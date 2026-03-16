import sys
import traceback
from typing import Any, Iterable

_printed_strs: set[str] = set()


def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred
    return "".join(tb_lines)


def print_once(data: Any):
    if hasattr(data, "__str__"):
        print_str = str(data)
    else:
        print_str = repr(data)
    if print_str not in _printed_strs:
        _printed_strs.add(print_str)
        print(print_str)


def merge_param_names(
    param_names: list[str] | Iterable[str], layers: int, stop_at_numbers: bool
) -> list[str]:
    """
    Merge the parameter names if they share the same prefix up to the given number of layers.
    If stop_at_numbers is True, the merging process will stop when meeting numbers (usually layer numbers).
    """
    merged_param_names = []
    for param_name in param_names:
        if stop_at_numbers:
            layer_names = param_name.split(".")
            current_layer = 0
            while current_layer < len(layer_names):
                if layer_names[current_layer].isdigit():
                    break
                current_layer += 1
            if current_layer == 0:
                # The first layer is a number
                continue
            param_name = ".".join(layer_names[:current_layer])

        if param_name in merged_param_names:
            continue

        if len(param_name.split(".")) <= layers:
            merged_param_names.append(param_name)
        else:
            param_prefix = ".".join(param_name.split(".")[:layers])
            if param_prefix not in merged_param_names:
                merged_param_names.append(param_prefix)
    return merged_param_names


def print_step_log(step_log: dict[str, Any]) -> None:
    for key, value in step_log.items():
        value = float(value)
        print(f"\t{key}: {value:.5f}")