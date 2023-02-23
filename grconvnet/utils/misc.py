import logging
from pathlib import Path

import torch

from grconvnet.models.grconvnet import GenerativeResnet


def get_root_dir() -> Path:
    return Path(__file__).parent.parent


def exists_in_subfolder(path: Path, subfolder: Path) -> Path:
    path = Path(path)

    if not path.exists():
        path = subfolder / path

    if not path.exists():
        raise FileNotFoundError(f"Model path {path} does not exist.")

    return path


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
