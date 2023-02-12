from typing import Any, Dict, Union
from pathlib import Path
import json
from dataclasses import is_dataclass, asdict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from matplotlib.figure import Figure


# MAX_JSON_EXPORT_ELEMENTS = 10_000


class ExtendedJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, torch.Tensor):
            # assert torch.numel(o) <= MAX_JSON_EXPORT_ELEMENTS
            return o.tolist()
        elif isinstance(o, np.ndarray):
            # assert o.size <= MAX_JSON_EXPORT_ELEMENTS
            return o.tolist()
        elif is_dataclass(o):
            return asdict(o)
        return json.JSONEncoder.default(self, o)


class Exporter:
    def __init__(
        self,
        export_dir: Path = None,
        convert_tensor_to_image: bool = True,
        max_json_elements: int = 10_000,
        json_encoder: json.JSONEncoder = None,
    ):
        """Converts data to a saveable format and saves it to disk. The data must be a
        dictionary or a dataclass. The keys of the dictionary or the fields of the
        dataclass must be strings. The values can be any of the following types:
        - torch.Tensor
        - numpy.ndarray
        - dataclasses.dataclass
        - json serializable types (int, float, str, list, dict, tuple, bool, NoneType)
        Tensors and numpy arrays are saved as .npy files. If the tensor is a 3-channel chw
        image with more than 20x20 pixels, it is saved as a .png file. If the tensor
        has more than max_json_elements elements, it is saved as a .npy file. If the
        tensor has less than max_json_elements elements, it is saved into a list in json file.
        Note that for nested structures, the types of the nested elements must be
        json serializable and are not converted to images etc.
        All results are saved in a subfolder of the export_dir folder.
        The name of the subfolder can be specified if a field "_export_name" is
        present in the data to export.

        Args:
            export_dir (Path, optional): The dictionairy where the results are saved to.
                Defaults to the "result" folder in this project.
            convert_tensor_to_image (bool, optional): Whether 3-channler chw or whc tensors
                with more than 20x20 pixels should be stored as .png instead of saving
                as .npy. Defaults to True.
            max_json_elements (int, optional): The maximum number of elements in a tensor
                that can be exported to json. Otherwise they get saved as .npy.
                Defaults to 10_000.
            json_encoder (json.JSONEncoder, optional): The json encoder to use for
                data that is not covered by the image and numpy export. Defaults to the
                ExtendedJsonEncoder.
        """
        if export_dir is None:
            export_dir = Path(__file__).parent.parent / "results"
        self.export_dir = export_dir

        self.json_encoder = json_encoder
        if self.json_encoder is None:
            self.json_encoder = ExtendedJsonEncoder

        self.convert_tensor_to_image = convert_tensor_to_image
        self.max_json_elements = max_json_elements

        self.i = 0

    def _image_convert(self, value):
        if not (
            isinstance(value, (torch.Tensor, np.ndarray))
            and value.ndim == 3
            and (
                (value.shape[0] == 3 and value.shape[1] > 20 and value.shape[2] > 20)
                or (value.shape[0] > 20 and value.shape[1] > 20 and value.shape[2] == 3)
            )
        ):
            return value

        if isinstance(value, torch.Tensor):
            value = value.numpy()

        # if pil_to_image gets a numpy array it assumes it is in hwc format
        # chw to hwc
        if value.shape[0] == 3:
            value = value.transpose(1, 2, 0)

        # FIXME: this is a hack to get the image to be saved as png
        value = value.astype(np.uint8)
        # value = Image.fromarray(value)
        value = F.to_pil_image(value)

        return value

    def __call__(self, data: Union["Dataclass", Dict[str, Any]]) -> Path:
        name = data.pop("_export_name", None)
        if name is None:
            self.i += 1
            name = f"result_{self.i}"

        sample_path = self.export_dir / name
        sample_path.mkdir(parents=True, exist_ok=True)

        if is_dataclass(data):
            data = asdict(data)

        json_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()

            if self.convert_tensor_to_image:
                value = self._image_convert(value)

            if isinstance(value, np.ndarray) and value.size <= self.max_json_elements:
                value = value.tolist()

            if isinstance(value, Image.Image):
                value.save(sample_path / f"{key}.png")
            elif isinstance(value, np.ndarray):
                np.save(sample_path / f"{key}.npy", value)
            elif isinstance(value, Figure):
                value.savefig(sample_path / f"{key}.png")
            else:
                json_data[key] = value

        with open(sample_path / "data.json", "w") as f:
            json.dump(json_data, f, cls=self.json_encoder, indent=4)

        return sample_path
