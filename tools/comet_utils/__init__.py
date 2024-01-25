import os
import yaml
import comet_ml
import easydict
import io
import torch
import contextlib
import logging

def get_binary_asset(experiment : str | comet_ml.APIExperiment, filename : str):
    if type(experiment) is str:
        api : comet_ml.API = comet_ml.API(api_key=os.environ["COMET_API_KEY"])
        apiexperiment : comet_ml.APIExperiment = api.get_experiment("electric-turtle", "mtr-deepracing", experiment)
    else:
        apiexperiment = experiment
    assetlist = apiexperiment.get_asset_list()
    assetid=None
    for asset in assetlist:
        if filename==asset["fileName"]:
            assetid=asset["assetId"]
            break
    asset = apiexperiment.get_asset(assetid, return_type="binary")
    return asset


def get_yaml_asset(experiment : str | comet_ml.APIExperiment, filename : str, raw=False):
    if type(experiment) is str:
        api : comet_ml.API = comet_ml.API(api_key=os.environ["COMET_API_KEY"])
        apiexperiment : comet_ml.APIExperiment = api.get_experiment("electric-turtle", "mtr-deepracing", experiment)
    else:
        apiexperiment = experiment
    assetlist = apiexperiment.get_asset_list()
    assetid=None
    for asset in assetlist:
        if filename==asset["fileName"]:
            assetid=asset["assetId"]
            break
    asset = apiexperiment.get_asset(assetid, return_type="binary")
    yamlstring = str(asset, encoding="ascii")
    if raw:
        return yamlstring
    return yaml.safe_load(yamlstring)

def get_statedict_asset(experiment : str | comet_ml.APIExperiment, filename : str, map_location : torch.device | str = "cpu"):
    binaryasset = get_binary_asset(experiment, filename)
    bytesio = io.BytesIO(binaryasset)
    return torch.load(bytesio, map_location=map_location)

def most_recent_model(experiment : str | comet_ml.APIExperiment, map_location : torch.device | str = "cpu", logger : logging.Logger | None = None):
    if type(experiment) is str:
        api : comet_ml.API = comet_ml.API(api_key=os.environ["COMET_API_KEY"])
        apiexperiment : comet_ml.APIExperiment = api.get_experiment("electric-turtle", "mtr-deepracing", experiment)
    else:
        apiexperiment = experiment
    # apiexperiment.get_parameters_summary(parameter="curr_epoch")
    assetlist = apiexperiment.get_asset_list()
    netassets = [a for a in assetlist if "model_" in a['fileName']]
    sortednetassets = sorted(netassets, key=lambda a : a["step"])

    model_file : str = (sortednetassets[-1])["fileName"]
    optimizer_file = model_file.replace("model_", "optimizer_")
    checkpoint_epoch = int(model_file.replace("model_epoch_","").replace(".pt",""))
    

    
    if logger is not None:
        logger.info("Getting model state dict")
    model_state_dict = get_statedict_asset(apiexperiment, model_file, map_location=map_location)
    if logger is not None:
        logger.info("Getting optimizer state dict")
    optimizer_state_dict = get_statedict_asset(apiexperiment, optimizer_file, map_location=map_location)
    return model_state_dict, optimizer_state_dict, checkpoint_epoch+1

def config_from_comet(experiment : str | comet_ml.APIExperiment):
    if type(experiment) is str:
        api : comet_ml.API = comet_ml.API(api_key=os.environ["COMET_API_KEY"])
        apiexperiment : comet_ml.APIExperiment = api.get_experiment("electric-turtle", "mtr-deepracing", experiment)
    else:
        apiexperiment = experiment
    configdict = get_yaml_asset(apiexperiment, "config.yaml")
    commandlineargs = get_yaml_asset(apiexperiment, "command_line_args.yaml")
    return configdict, commandlineargs

def generate_experiment(experimentname : str | None, cfg : easydict.EasyDict, cfg_file : str | None = None):
    localrank = cfg.LOCAL_RANK
    if not (localrank==0):
        return None, ["",]
    api_key=os.environ["COMET_API_KEY"]
    if experimentname is None:
        comet_experiment = comet_ml.Experiment(api_key=api_key, workspace="electric-turtle", project_name="mtr-deepracing", auto_output_logging="simple")
        if cfg_file is not None:
            comet_experiment.log_asset(cfg_file, file_name="config.yaml", overwrite=True, copy_to_tmp=False)
        clusterfile = os.path.join(cfg.ROOT_DIR, cfg["MODEL"]["MOTION_DECODER"]["INTENTION_POINTS_FILE"])
        comet_experiment.log_asset(clusterfile, file_name="clusters.pkl", overwrite=True, copy_to_tmp=False)
    else:
        api : comet_ml.API = comet_ml.API(api_key=api_key)
        apiexperiment : comet_ml.APIExperiment = api.get_experiment("electric-turtle", "mtr-deepracing", experimentname)
        comet_experiment : comet_ml.ExistingExperiment = comet_ml.ExistingExperiment(api_key=api_key, workspace=apiexperiment.workspace, project_name=apiexperiment.project_name, experiment_key=apiexperiment.key, auto_output_logging="simple")
        
    return comet_experiment, ["_" + comet_experiment.get_name(),]

def local_context(comet_experiment : comet_ml.Experiment | None, stage : str = "train"):
    if comet_experiment is None:
        return contextlib.nullcontext()
    if stage=="train":
        return comet_experiment.train()
    elif (stage=="eval"):
        return comet_experiment.validate()
    elif (stage=="test"):
        return comet_experiment.test()
    else:
        raise ValueError("\"stage\" must be one of \{train, test, eval\}")
    

    
