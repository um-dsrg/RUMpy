from rumpy.SISR.models.interface import SISRInterface
import os


def prep_models(model_loc, experiment_names, eval_epochs, gpu, scale=4, sp_gpu=0):
    """
    Initializes and sets up specified models for evaluation.
    :param model_loc: Location from which to extract saved models
    :param experiment_names: List or Tuple of experiment names in the SISR results folder.
    :param eval_epochs: List of specific epochs to evaluate for each model.
    :param gpu: Specify whether to use a GPU in a computation.
    :param scale: super-resolution model scale (restricted on certain models)
    :return: model bundles (dict) & empty metrics (dict)
    """
    models = []
    for experiment, eval_epoch in zip(experiment_names, eval_epochs):

        if os.path.isdir(experiment):
            base_loc = os.path.dirname(experiment)
            local_name = os.path.basename(experiment)
        else:
            base_loc = model_loc
            local_name = experiment

        models.append(
            SISRInterface(base_loc, local_name,
                          load_epoch=int(eval_epoch) if eval_epoch.isnumeric() else eval_epoch,
                          gpu='off' if not gpu else 'single', scale=scale, sp_gpu=sp_gpu))
    return models


def minimize_model(base_model_loc, model_and_epoch):
    experiment_names, eval_epochs = zip(*model_and_epoch)  # unpacking model info
    models = prep_models(base_model_loc, experiment_names, eval_epochs, False)
    for model in models:
        model.save(minimal=True)
