import logging
from pathlib import Path
from typing import Dict, List

import torch

from grconvnet.models.grconvnet import GenerativeResnet

MODELS_PATH = Path(__file__).parent.parent / "trained-models-jit"


def get_model_path(model_name: str, version_name: str):
    return MODELS_PATH / model_name / version_name


def list_pretrained_models() -> Dict[str, List[str]]:
    """Get an overview about all exisitng pretrained models.
    The result is an Dictionairy where all keys are the name of the model/the corresponding
    folder and the keys are a list of file names which give the name of the model
    versions containes in the model folder.

    Returns:
        Dict[str, List[str]]: All models and their versions.
    """
    models = {
        model_dir.name: [f.name for f in model_dir.iterdir() if f.suffix == ".pt"]
        for model_dir in MODELS_PATH.iterdir()
        if model_dir.is_dir()
    }

    return models


def load_pretrained(
    name: str = "cornell-randsplit-rgbd-grconvnet3-drop1-ch32",
    version: str = None,
) -> torch.nn.Module:
    """Loads one of the pretrained models as a pytorch Model class.
    This uses the jit loading option so the model can be used independently of the
    repository structure. However a grconvnet3 class is instantiated afterwards
    and setup with the weidgts from the loaded network in order to make the
    network runnable.

    Args:
        name (str, optional): The name od the model to load. This must be a name
            of a subfolder in the 'trained-models' folder (= a kay from the dict
            obtained by list_pretrained_models()).
            Defaults to "cornell-randsplit-rgbd-grconvnet3-drop1-ch32".
        version (str, optional): The name of the version of the model.
            This must be a name of a model file contained in the model subfolder.
            Defaults to the first one in the folder sorted alphabetically.

    Returns:
        torch.nn.Module: The loaded pytorch module.
    """
    model_dir = MODELS_PATH / name

    if version is None:
        versions = [f.name for f in model_dir.iterdir() if f.suffix == ".pt"]
        version = sorted(versions)[0]

    model_path = model_dir / version

    model = torch.jit.load(model_path)

    new_model = GenerativeResnet()
    new_model.load_state_dict(model.state_dict())
    model = new_model

    return model


def convert_models():
    """An added utility script which tries to load the pickled pretrained models and
    saves them to a serialized version.
    """
    logging.info(torch.__version__)

    for model_dir in (Path(__file__).parent / "trained-models").iterdir():
        if not model_dir.is_dir():
            continue
        for model_path in model_dir.iterdir():
            if model_path.suffix == ".txt":
                continue

            logging.info(f"Loading {model_path}...")
            model = torch.load(model_path)

            if model_dir.name.startswith("cornell"):
                if model_dir.name.endswith("16"):
                    new_model = GenerativeResnet(channel_size=16)
                else:
                    new_model = GenerativeResnet()

                new_model.load_state_dict(model.state_dict())
                model = new_model

            logging.info(f"Saving converted {model_path}...")
            destination_path = (
                Path(__file__).parent
                / "trained-models-jit"
                / model_dir.name
                / f"{model_path.name}.pt"
            )
            torch.jit.script(model).save(destination_path)
