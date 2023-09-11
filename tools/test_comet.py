import sys
import os
this_dir = os.path.dirname(__file__)
sys.path = [os.path.dirname(this_dir),] + sys.path
import mtr.datasets, mtr.models.model
import yaml
import easydict
import logging
import torch, torch.utils.data as torch_data, torch.nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure, matplotlib.axes
import shutil
import time
import tqdm

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

def test_model(comet_experiment : str, tempdir : str, batch_size : int, save_every : int, with_tqdm : bool):
    fp = os.path.join(this_dir, "cfgs", "deepracing", "mtr+100_percent_data.yaml" )
    with open(fp, "r") as f:
        config = easydict.EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    print(config)

    logger = create_logger()
    dist = False
    val_set, val_loader, val_sampler = \
        mtr.datasets.build_dataloader(config["DATA_CONFIG"], batch_size, dist, 
                                    training=False, workers=0, logger=logger, total_epochs=1)
    num_samples = len(val_set)
    # train_loader = torch_data.DataLoader(dset, batch_size=1)
    # dataloader_iter = iter(train_loader)
    # batch = next(dataloader_iter)
    # print('new iters')
    # print(batch)
    gpu_index = 0
    plotdir = os.path.normpath(os.path.join(this_dir, "..", "output", "test_plots"))
    if os.path.isdir(plotdir):
        shutil.rmtree(plotdir)
    os.makedirs(plotdir)
    model_dir = os.path.join(tempdir, "default_%s" % (comet_experiment,), "ckpt")

    # model_file = os.path.join(model_dir, "model_comet_epoch_10.pt")
    model_file = os.path.join(model_dir, "latest_model.pth")
    with open(model_file, "rb") as f:
        ckpt = torch.load(f)
        # state_dict = ckpt
        state_dict = ckpt["model_state"]
    
    model = mtr.models.model.MotionTransformer(config["MODEL"]).eval()
    model.load_state_dict(state_dict)
    model.cuda(gpu_index)
    first_param : torch.nn.Parameter = next(model.parameters())
    min_ade_array = torch.empty(num_samples, dtype=first_param.dtype, device=first_param.device)
    min_ade_index_array = torch.empty(num_samples, dtype=torch.int64, device=first_param.device)
    computation_time_list = []
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
    for (i, batch_dict) in tq:
        input_dict_keys = batch_dict["input_dict"].keys()
        # print()
        # print(input_dict_keys)
        for key in input_dict_keys:
            input_type = type(batch_dict["input_dict"][key])
            if input_type is torch.Tensor:
                batch_dict["input_dict"][key] = batch_dict["input_dict"][key].cuda(gpu_index)
        input_dict : dict = batch_dict["input_dict"]
        # print(input_dict["obj_trajs_future_mask"][0], input_dict["obj_trajs_future_mask"][0].shape)
        # print(input_dict["obj_trajs_pos"][0], input_dict["obj_trajs_pos"][0].shape)
        # print(input_dict["obj_trajs_last_pos"][0], input_dict["obj_trajs_last_pos"][0].shape)
        # print()
        tick = time.time()
        with torch.no_grad():
            batch_pred_dicts : dict = model(batch_dict)
        tock = time.time()
        computation_time = tock - tick
        history_torch : torch.Tensor = input_dict["obj_trajs_pos"][:,0,:,[0,1]]
        # print()
        # print(history_torch[0])
        # print()
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
        else:
            torch.min(ade, dim=-1, out=(min_ade_array[array_idx:array_idx_end], min_ade_index_array[array_idx:array_idx_end]))
        array_idx=array_idx_end

        computation_time_list.append(computation_time)
        if save_every>0 and ((i%save_every)==0):
            fig : matplotlib.figure.Figure = plt.figure()
            mode_color = "black"
            for j in range(pred_traj_torch.shape[1]-1):
                plt.plot(pred_traj_torch[-1,j,:,0].cpu(), pred_traj_torch[-1,j,:,1].cpu(), color=mode_color, alpha=0.25)
            plt.plot(pred_traj_torch[-1,-1,:,0].cpu(), pred_traj_torch[-1,-1,:,1].cpu(), color=mode_color, label="Predicted Modes", alpha=0.25)
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
                    "computation_time" : computation_time_list[-1]
                }, f, Dumper=yaml.SafeDumper)
    computation_time_array : np.ndarray = np.asarray(computation_time_list)
    # min_ade_array : np.ndarray = np.asarray(min_ade_list)
    with open(os.path.join(plotdir, "summary.yaml"), "w") as f:
        yaml.dump({
            "min_ade" : torch.mean(min_ade_array).item(),
            "min_ade_stdev" : torch.std(min_ade_array).item(),
            "computation_time" : float(np.mean(computation_time_array)),
            "computation_time_stdev" : float(np.std(computation_time_array)),
        }, f, Dumper=yaml.SafeDumper)

        # exit(0)
if __name__=="__main__":
    import argparse
    print("Hello comet test!")
    parser : argparse.ArgumentParser = argparse.ArgumentParser(prog="mtr_comet_test", description="Test dat MTR from comet")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--tempdir", type=str, default="/bigtemp/ttw2xk/comet_dump/MTR/deepracing/mtr+100_percent_data")
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tqdm", action="store_true")
    args = parser.parse_args()

    test_model(args.experiment, args.tempdir, args.batch_size, args.save_every, args.tqdm)