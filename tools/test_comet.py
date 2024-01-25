import sys
import os
this_dir = os.path.dirname(__file__)
sys.path = [os.path.dirname(this_dir),] + sys.path
import mtr.datasets, mtr.models.model
import yaml
import easydict
import logging
import torch, torch.utils.data as torchdata, torch.nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure, matplotlib.axes
import shutil
import time
import tqdm
import deepracing_models.math_utils
import comet_ml
from comet_utils import get_yaml_asset
import io

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
def test_model(model : mtr.models.model.MotionTransformer, 
               val_loader : torchdata.DataLoader, 
               plotdir : str, 
               save_every = -1, with_tqdm = True):
    # if os.path.isdir(plotdir):
    #     shutil.rmtree(plotdir)
    os.makedirs(plotdir, exist_ok=True)
    num_samples = len(val_loader.dataset)
    batch_size = val_loader.batch_size
    first_param : torch.nn.Parameter = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype
    scores_array = torch.empty([num_samples, 6], dtype=dtype, device=device)
    history_array = torch.empty([num_samples, 11, 2], dtype=dtype, device=device)
    ground_truth_array = torch.empty([num_samples, 80, 2], dtype=dtype, device=device)
    predictions_array = torch.empty([num_samples, 6, 80, 2], dtype=dtype, device=device)
    min_ade_array = torch.empty(num_samples, dtype=dtype, device=device)
    min_ade_index_array = torch.empty(num_samples, dtype=torch.int64, device=device)
    lateral_error_array = torch.empty(num_samples, dtype=dtype, device=device)
    longitudinal_error_array = torch.empty(num_samples, dtype=dtype, device=device)
    # computation_time_list = []
    for (i, batch_dict) in enumerate(val_loader):
        batch_pred_dicts : dict = model(batch_dict)
        if i>10:
            break
    loader_enum = enumerate(val_loader)
    if with_tqdm:
        tq = tqdm.tqdm(loader_enum, total = int(np.ceil(num_samples/batch_size)))
    else:
        tq = loader_enum
    array_idx = 0
    # lateral_error_list = []
    computation_time_list = []
    with torch.no_grad():
        for (i, batch_dict) in tq:
            input_dict_keys = batch_dict["input_dict"].keys()


            for key in input_dict_keys:
                input_type = type(batch_dict["input_dict"][key])
                if input_type is torch.Tensor:
                    batch_dict["input_dict"][key] = batch_dict["input_dict"][key].to(device=device)
            input_dict : dict = batch_dict["input_dict"]
            tick = time.time()
            batch_pred_dicts : dict = model(batch_dict)
            # print(batch_pred_dicts.keys())
            # print(batch_pred_dicts["input_dict"].keys())
            # torch.set_printoptions()[]
            # print(batch_pred_dicts["input_dict"]["map_polylines_center"][-1])
            # print(batch_pred_dicts["input_dict"]["map_polylines_center"][-1].shape)
            # print(batch_pred_dicts["input_dict"]["map_polylines"][0,:,0,[0,1]])
            # print(batch_pred_dicts["input_dict"]["map_polylines"][0,:,1,[0,1]])
            # print(batch_pred_dicts["input_dict"]["map_polylines"][0,:,0,[0,1]].shape)
            # print(batch_pred_dicts["input_dict"]["map_polylines"][0].shape)
            # print(batch_pred_dicts["input_dict"]["map_polylines_mask"][0].shape)
            tock = time.time()
            computation_time_list.append(float(tock-tick))
            if with_tqdm:
                tq.set_postfix({"computation_time" : float(computation_time_list[-1])})
            history_torch : torch.Tensor = input_dict["obj_trajs_pos"][:,0,:,[0,1]]
            center_gt_trajs_torch : torch.Tensor = input_dict["center_gt_trajs"][:,:,[0,1]]
            pred_traj_torch : torch.Tensor = batch_pred_dicts["pred_trajs"][:,:,:,[0,1]]

            pred_scores_torch : torch.Tensor = batch_pred_dicts["pred_scores"]
            displacements_torch : torch.Tensor = pred_traj_torch - center_gt_trajs_torch[:,None]
            displacement_error = torch.norm(displacements_torch, p=2.0, dim=-1)
            ade : torch.Tensor = torch.mean(displacement_error, dim=-1)
            batchdim = center_gt_trajs_torch.shape[0]
            array_idx_end = array_idx+batchdim
            if array_idx_end>=min_ade_array.shape[0]:
                torch.min(ade, dim=-1, out=(min_ade_array[array_idx:], min_ade_index_array[array_idx:]))
                best_mode_idx = min_ade_index_array[array_idx:]
            else:
                torch.min(ade, dim=-1, out=(min_ade_array[array_idx:array_idx_end], min_ade_index_array[array_idx:array_idx_end]))
                best_mode_idx = min_ade_index_array[array_idx:array_idx_end]
            best_modes = torch.stack([ pred_traj_torch[ j, best_mode_idx[j] ] for j in range(batchdim)], dim=0)
            predictions_array[array_idx:array_idx_end] = pred_traj_torch[:]
            scores_array[array_idx:array_idx_end] = pred_scores_torch[:]
            ground_truth_array[array_idx:array_idx_end] = center_gt_trajs_torch[:]
            history_array[array_idx:array_idx_end] = history_torch[:]

            batch_dim = center_gt_trajs_torch.shape[0]
            Nfuture = center_gt_trajs_torch.shape[1]
            tsamp = torch.linspace(0.0, 3.0, steps=Nfuture, dtype=center_gt_trajs_torch.dtype, device=center_gt_trajs_torch.device)
            ssamp = tsamp/tsamp[-1]
            ssamp_batch = ssamp.unsqueeze(0).expand(batch_dim, Nfuture)

            kfit = 7
            _, curves_gt = deepracing_models.math_utils.bezierLsqfit(best_modes, kfit, t=ssamp_batch)
        
            Mderiv = deepracing_models.math_utils.bezierM(ssamp_batch, kfit-1)
            curve_derivs_gt = kfit*(curves_gt[:,1:] - curves_gt[:,:-1])/tsamp[-1]

            curve_vels = torch.matmul(Mderiv, curve_derivs_gt)

            curve_tangents = curve_vels/torch.norm(curve_vels, p=2.0, dim=-1, keepdim=True)
            curve_normals = curve_tangents[:,:,[1,0]].clone()
            curve_normals[:,:,0]*=-1.0
            
            rotmats_decomposition = torch.stack([curve_tangents, curve_normals], axis=3).transpose(-2,-1)
            translations_decomposition = torch.matmul(rotmats_decomposition, -center_gt_trajs_torch.unsqueeze(-1)).squeeze(-1)

            decompositions = torch.matmul(rotmats_decomposition, best_modes.unsqueeze(-1)).squeeze(-1) + translations_decomposition
            
            longitudinal_errors = torch.abs(decompositions[:,:,0])
            lateral_errors = torch.abs(decompositions[:,:,1])
            if array_idx_end>=lateral_error_array.shape[0]:
                torch.mean(longitudinal_errors, dim=1, out=longitudinal_error_array[array_idx:])
                torch.mean(lateral_errors, dim=1, out=lateral_error_array[array_idx:])
            else:
                torch.mean(longitudinal_errors, dim=1, out=longitudinal_error_array[array_idx:array_idx_end])
                torch.mean(lateral_errors, dim=1, out=lateral_error_array[array_idx:array_idx_end])
            array_idx=array_idx_end

            # computation_time_list.append(computation_time)
            if save_every>0 and ((i%save_every)==0):
                bounds_cpu : torch.Tensor = batch_pred_dicts["input_dict"]["map_polylines_center"][-1,0:-3,[0,1]].cpu()

                fig : matplotlib.figure.Figure = plt.figure()
                mode_color = "black"
                for j in range(pred_traj_torch.shape[1]-1):
                    plt.plot(pred_traj_torch[-1,j,:,0].cpu(), pred_traj_torch[-1,j,:,1].cpu(), color=mode_color, alpha=0.25)
                plt.plot(pred_traj_torch[-1,-1,:,0].cpu(), pred_traj_torch[-1,-1,:,1].cpu(), color=mode_color, label="Predicted Modes", alpha=0.25)
                plt.scatter(bounds_cpu[:,0].cpu(), bounds_cpu[:,1].cpu(), label="Bounds", s=0.1, alpha=0.5)
                plt.scatter(history_torch[-1,:,0].cpu(), history_torch[-1,:,1].cpu(), s=0.1, c="blue", label="History")
                plt.scatter(center_gt_trajs_torch[-1,:,0].cpu(), center_gt_trajs_torch[-1,:,1].cpu(), s=0.1, c="green", label="Ground Truth")
                plt.legend()
                fig.savefig(os.path.join(plotdir, "fig_%d.png" % (i,)))
                fig.savefig(os.path.join(plotdir, "fig_%d.pdf" % (i,)))
                plt.close()
                with open(os.path.join(plotdir, "fig_%d.yaml" % (i,)), "w") as f:
                    yaml.dump({
                        "ade" : ade[-1].tolist(),
                        "scores" : pred_scores_torch[-1].tolist(),
                        "min_ade" : min_ade_array[array_idx_end-1].item(),
                        "lateral_error" : lateral_error_array[array_idx_end-1].item(),
                        "longitudinal_error" : longitudinal_error_array[array_idx_end-1].item()
                    }, f, Dumper=yaml.SafeDumper)
    computation_time_array = np.asarray(computation_time_list, dtype=np.float64)
    with open(os.path.join(plotdir, "summary.yaml"), "w") as f:
        yaml.dump({
            "min_ade" : torch.mean(min_ade_array).item(),
            "min_ade_stdev" : torch.std(min_ade_array).item(),
            "lateral_error" : torch.mean(lateral_error_array).item(),
            "longitudinal_error" : torch.mean(longitudinal_error_array).item(),
            "computation_time" : float(np.mean(computation_time_array))
        }, f, Dumper=yaml.SafeDumper)
    with open(os.path.join(plotdir, "data.npz"), "wb") as f:
        np.savez(f, **{
            "scores" : scores_array.cpu().numpy(),
            "best_curve_idx" : min_ade_index_array.cpu().numpy(),
            "history" : history_array.cpu().numpy(),
            "ground_truth" : ground_truth_array.cpu().numpy(),
            "predictions" : predictions_array.cpu().numpy(),
            "lateral_error" : lateral_error_array.cpu().numpy(),
            "longitudinal_error" : longitudinal_error_array.cpu().numpy(),
            "ade" : min_ade_array.cpu().numpy(),
            "computation_time" : computation_time_array
        })
    return min_ade_array.cpu(), lateral_error_array, longitudinal_error_array
def go(comet_experiment : str, batch_size : int, save_every : int, with_tqdm : bool, gpu_index : int, results_dir : str):
    torch.cuda.set_device(torch.device("cuda:%d" % (gpu_index,)))
    api : comet_ml.API = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
    api_experiment : comet_ml.APIExperiment = api.get(workspace="electric-turtle", project_name="mtr-deepracing", experiment=comet_experiment)
    
    asset_list = api_experiment.get_asset_list()
    net_assets = []
    config_asset = None
    for asset in asset_list:
        if "model_" in asset["fileName"]:
            net_assets.append(asset)
        elif asset["fileName"]=="config.yaml":
            config_asset = asset
    net_assets = sorted(net_assets, key=lambda a : a["step"])
    configstring = str(api_experiment.get_asset(config_asset["assetId"], return_type="binary"), encoding="ascii")
    config = easydict.EasyDict(yaml.safe_load(io.StringIO(configstring)))
    config["DATA_CONFIG"]["SPLIT_DIR"]["test"] = "processed_scenarios_test"
    config["DATA_CONFIG"]["INFO_FILE"]["test"] = "processed_scenarios_test_infos.pkl"
    config["DATA_CONFIG"]["SAMPLE_INTERVAL"]["test"] = 1

    logger = create_logger(log_file = os.path.join(results_dir, "log.txt"))
    dist = False

    net_binary = api_experiment.get_asset(net_assets[-1]["assetId"], return_type="binary")
    net_bytesio = io.BytesIO(net_binary)
    model = mtr.models.model.MotionTransformer(config["MODEL"]).eval()
    model.load_state_dict(torch.load(net_bytesio, map_location="cpu"))
    model.cuda()
    main_plot_dir = os.path.join(results_dir, api_experiment.get_name())

    plot_dir = os.path.join(main_plot_dir, "test_plots")
    val_set, val_loader, val_sampler = \
        mtr.datasets.build_dataloader(config["DATA_CONFIG"], batch_size, dist, 
                                    training=False, workers=30, logger=logger, total_epochs=1)
    min_ade_array, lateral_error_array, longitudinal_error_array = \
        test_model(model, val_loader, plot_dir, save_every=save_every, with_tqdm=with_tqdm)
    
    ade_sort = torch.argsort(min_ade_array, descending=True)
    plot_dir = os.path.join(main_plot_dir, "test_plots_top100_ade")
    val_set, val_loader, val_sampler = \
        mtr.datasets.build_dataloader(config["DATA_CONFIG"], 1, dist, subset_indices=ade_sort[:100],
                                    training=False, workers=0, logger=logger, total_epochs=1)
    test_model(model, val_loader, plot_dir, save_every=1, with_tqdm=with_tqdm)


    lateral_error_sort = torch.argsort(lateral_error_array, descending=True)
    plot_dir = os.path.join(main_plot_dir, "test_plots_top100_lateral_error")
    val_set, val_loader, val_sampler = \
        mtr.datasets.build_dataloader(config["DATA_CONFIG"], 1, dist, subset_indices=lateral_error_sort[:100],
                                    training=False, workers=0, logger=logger, total_epochs=1)
    test_model(model, val_loader, plot_dir, save_every=1, with_tqdm=with_tqdm)


    long_error_sort = torch.argsort(longitudinal_error_array, descending=True)
    plot_dir = os.path.join(main_plot_dir, "test_plots_top100_longitudinal_error")
    val_set, val_loader, val_sampler = \
        mtr.datasets.build_dataloader(config["DATA_CONFIG"], 1, dist, subset_indices=long_error_sort[:100],
                                    training=False, workers=0, logger=logger, total_epochs=1)
    test_model(model, val_loader, plot_dir, save_every=1, with_tqdm=with_tqdm)

        # exit(0)
if __name__=="__main__":
    import argparse
    print("Hello comet test!")
    parser : argparse.ArgumentParser = argparse.ArgumentParser(prog="mtr_comet_test", 
                                                               description="Test dat MTR from comet",
                                                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--results-dir", type=str, default="/p/DeepRacing/mtr_results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tqdm", action="store_true")
    args = parser.parse_args()
    go(args.experiment, args.batch_size, args.save_every, args.tqdm, args.gpu, args.results_dir)