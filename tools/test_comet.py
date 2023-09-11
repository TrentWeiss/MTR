import sys
import os
this_dir = os.path.dirname(__file__)
sys.path = [os.path.dirname(this_dir),] + sys.path
import mtr.datasets, mtr.models.model
import yaml
import easydict
import logging
import torch, torch.utils.data as torch_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure, matplotlib.axes
import shutil
import time

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

def test_model(comet_experiment : str, tempdir : str, save_all : bool):
    fp = os.path.join(this_dir, "cfgs", "deepracing", "mtr+100_percent_data.yaml" )
    with open(fp, "r") as f:
        config = easydict.EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    print(config)

    logger = create_logger()
    batch_size = 1
    train_set, train_loader, train_sampler = \
        mtr.datasets.build_dataloader(config["DATA_CONFIG"], batch_size, False, 
                                    training=False, workers=0, logger=logger, total_epochs=1)
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

    min_ade_list = []
    computation_time_list = []
    for (i, batch_dict) in enumerate(train_loader):
        batch_pred_dicts : dict = model(batch_dict)
        if i>10:
            break
    for (i, batch_dict) in enumerate(train_loader):
        # print(batch_dict.keys())
        input_dict_keys = batch_dict["input_dict"].keys()
        for key in input_dict_keys:
            input_type = type(batch_dict["input_dict"][key])
            if input_type is torch.Tensor:
                batch_dict["input_dict"][key] = batch_dict["input_dict"][key].cuda(gpu_index)
        input_dict : dict = batch_dict["input_dict"]
        tick = time.time()
        with torch.no_grad():
            batch_pred_dicts : dict = model(batch_dict)
        tock = time.time()
        computation_time = tock - tick
        # print(batch_pred_dicts.keys())
        center_gt_trajs : np.ndarray = input_dict["center_gt_trajs"][0].cpu().numpy()[:,[0,1]]
        pred_traj : np.ndarray = batch_pred_dicts["pred_trajs"][0].cpu().numpy()[:,:,[0,1]]
        displacements = pred_traj - center_gt_trajs
        displacement_error = np.linalg.norm(displacements, ord=2.0, axis=-1)
        ade : np.ndarray = np.mean(displacement_error, axis=-1)
        scores : np.ndarray = batch_pred_dicts["pred_scores"][0].cpu().numpy()
        scores=scores/np.sum(scores)
        min_ade = float(np.min(ade))
        min_ade_list.append(min_ade)
        computation_time_list.append(computation_time)
        if save_all:
            fig : matplotlib.figure.Figure = plt.figure()
            for j in range(pred_traj.shape[0]):
                plt.plot(pred_traj[j,:,0], pred_traj[j,:,1])
            plt.scatter(center_gt_trajs[:,0], center_gt_trajs[:,1], s=0.1)
            fig.savefig(os.path.join(plotdir, "fig_%d.png" % (i,)))
            fig.savefig(os.path.join(plotdir, "fig_%d.pdf" % (i,)))
            plt.close()
            with open(os.path.join(plotdir, "fig_%d.yaml" % (i,)), "w") as f:
                yaml.dump({
                    "ade" : ade.tolist(),
                    "scores" : scores.tolist(),
                    "min_ade" : min_ade,
                    "computation_time" : computation_time
                }, f, Dumper=yaml.SafeDumper)
        computation_time_array : np.ndarray = np.asarray(computation_time_list)
        min_ade_array : np.ndarray = np.asarray(min_ade_list)
        with open(os.path.join(plotdir, "summary.yaml"), "w") as f:
            yaml.dump({
                "min_ade" : np.mean(min_ade_array),
                "min_ade_stdev" : np.std(min_ade_array),
                "computation_time" : np.mean(computation_time_array),
                "computation_time_stdev" : np.std(computation_time_array),
            }, f, Dumper=yaml.SafeDumper)

        # exit(0)
if __name__=="__main__":
    import argparse
    print("Hello comet test!")
    parser : argparse.ArgumentParser = argparse.ArgumentParser(prog="mtr_comet_test", description="Test dat MTR from comet")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--tempdir", type=str, default="/bigtemp/ttw2xk/comet_dump/MTR/deepracing/mtr+100_percent_data")
    parser.add_argument("--saveall", action="store_true")
    args = parser.parse_args()

    test_model(args.experiment, args.tempdir, args.saveall)