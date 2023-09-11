from typing import Dict, Any
from copy import deepcopy
import importlib


def module_from_config(config: Dict[str, Any]):
    config = deepcopy(config)

    import_path = config.pop("class").split(".")
    module_cls = getattr(
        importlib.import_module(".".join(import_path[:-1])), import_path[-1]
    )

    constructor_name = config.pop("constructor", None)
    if constructor_name is not None:
        module_cls = getattr(module_cls, constructor_name)

    module_kwargs = {}

    for arg_name, arg_value in config.items():
        if isinstance(arg_value, dict) and "class" in arg_value:
            submodule_config = arg_value
            submodule = module_from_config(submodule_config)
            module_kwargs[arg_name] = submodule
        else:
            module_kwargs[arg_name] = arg_value

    return module_cls(**module_kwargs)
