import numpy as np
import numpy.typing as npt
import cv2
from typing import Any, Callable, Union
from cv2.typing import MatLike


def resize_with_padding(
    img: npt.NDArray[Any], new_shape_hw: tuple[int, ...]
) -> npt.NDArray[Any]:
    """
    img: (..., C, H, W)

    new_shape_hw: (new_H, new_W)

    return: (..., C, new_H, new_W)
    """

    assert (
        len(new_shape_hw) == 2
    ), f"new_shape_hw must be a tuple of length 2, but got {new_shape_hw}"

    batch_shape = img.shape[:-3]
    C, H, W = img.shape[-3:]

    original_aspect_ratio = W / H
    new_H = new_shape_hw[0]
    new_W = new_shape_hw[1]
    new_aspect_ratio = new_W / new_H
    if original_aspect_ratio > new_aspect_ratio:
        # Pad upwards and downwards
        new_H_without_padding = int(new_W / original_aspect_ratio)
        new_W_without_padding = new_W
        padding_top = (new_H - new_H_without_padding) // 2
        padding_bottom = new_H - new_H_without_padding - padding_top
        padding_sequence = [(0, 0) for _ in range(len(batch_shape))] + [
            (0, 0),
            (padding_top, padding_bottom),
            (0, 0),
        ]

    else:
        # Pad left and right
        new_W_without_padding = int(new_H * original_aspect_ratio)
        new_H_without_padding = new_H
        padding_left = (new_W - new_W_without_padding) // 2
        padding_right = new_W - new_W_without_padding - padding_left
        padding_sequence = [(0, 0) for _ in range(len(batch_shape))] + [
            (0, 0),
            (0, 0),
            (padding_left, padding_right),
        ]

    img = img.reshape(-1, 3, H, W)
    resized_img = np.zeros(
        (img.shape[0], 3, new_H_without_padding, new_W_without_padding), dtype=img.dtype
    )
    for i in range(img.shape[0]):
        img_hwc = img[i].transpose(1, 2, 0)
        img_hwc = cv2.resize(img_hwc, (new_W_without_padding, new_H_without_padding))
        resized_img[i] = img_hwc.transpose(2, 0, 1)
    resized_img = resized_img.reshape(
        *batch_shape, C, new_H_without_padding, new_W_without_padding
    )
    resized_img = np.pad(
        resized_img, padding_sequence, mode="constant", constant_values=0
    )

    return resized_img


def resize_with_cropping(
    source_frame_hwc: npt.NDArray[Any],
    display_wh: tuple[int, int],
) -> npt.NDArray[Any]:
    """
    source_frame: (..., H, W, C)
    Crops and resizes a source frame to fit the display resolution while preserving aspect ratio.
    This logic is adapted directly from the user's robust BaseCamera implementation.
    """
    source_height, source_width = source_frame_hwc.shape[-3:-1]
    display_width, display_height = display_wh

    source_wh_ratio = source_width / source_height
    display_wh_ratio = display_width / display_height

    if source_wh_ratio > display_wh_ratio:
        # source is "wider" than display, crop the width
        new_width = int(source_height * display_wh_ratio)
        margin = (source_width - new_width) // 2
        cropped_frame = source_frame_hwc[..., :, margin : margin + new_width, :]
    else:
        # source is "taller" than or same as display, crop the height
        new_height = int(source_width / display_wh_ratio)
        margin = (source_height - new_height) // 2
        cropped_frame = source_frame_hwc[..., margin : margin + new_height, :, :]

    if len(source_frame_hwc.shape) == 4:
        resized_images = np.zeros(
            (
                source_frame_hwc.shape[0],
                display_height,
                display_width,
                source_frame_hwc.shape[-1],
            ),
            dtype=source_frame_hwc.dtype,
        )
        for i in range(source_frame_hwc.shape[0]):
            resized_images[i] = cv2.resize(cropped_frame[i], display_wh)
    else:
        resized_images = cv2.resize(cropped_frame, display_wh)
    resized_images = np.array(resized_images)
    return resized_images



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