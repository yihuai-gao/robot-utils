import numpy as np
import numpy.typing as npt
import cv2
from typing import Any, Callable, Union
from cv2.typing import MatLike



def aggregate_dict(
    dictionaries: list[dict[str, Any]] | dict[Any, dict[str, Any]],
    convert_to_numpy: bool,
    key_name: str = "",
) -> dict[str, Any]:
    """
    Aggregate a list of dictionaries or a dictionary of dictionaries into a single dictionary.
    """
    aggregated_dict: dict[str, Any] = {}
    if isinstance(dictionaries, list):
        for single_dict in dictionaries:
            for key, value in single_dict.items():
                if key not in aggregated_dict:
                    aggregated_dict[key] = []
                aggregated_dict[key].append(value)

    elif isinstance(dictionaries, dict):
        assert key_name != "", "Key name is required for dictionary aggregation"
        aggregated_dict[key_name] = []
        for key, value in dictionaries.items():
            assert (
                key_name not in value
            ), f"Key {key_name} is not allowed in the dictionary. Please rename the key for aggregation"
            aggregated_dict[key_name].append(key)
            for key, value in value.items():
                if key not in aggregated_dict:
                    aggregated_dict[key] = []
                aggregated_dict[key].append(value)

    for key, value in aggregated_dict.items():
        if key == key_name:
            continue
        if isinstance(value, list):
            if isinstance(value[0], dict):
                # Value is a list of dictionaries. Will be flattened in the next step
                value = aggregate_dict(value, convert_to_numpy)
            elif convert_to_numpy:
                if not isinstance(value[0], str):
                    try:
                        aggregated_dict[key] = np.array(value)
                    except Exception as e:
                        print(f"Error aggregating {key}: {e}")
                        for v in value:
                            print(f"{v.shape}")
                        raise e
    return aggregated_dict


def dict_apply(x: dict[str, Any], func: Callable[[Any], Any]) -> dict[str, Any]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result