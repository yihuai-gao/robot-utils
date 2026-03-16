import numpy as np
import numpy.typing as npt
import cv2
from typing import Any, Callable, Union
from cv2.typing import MatLike



def resize_frame_without_distortion(
    source_frame: npt.NDArray[np.uint8],
    source_wh: tuple[int, int],
    display_wh: tuple[int, int],
) -> MatLike:
    """
    Crops and resizes a source frame to fit the display resolution while preserving aspect ratio.
    This logic is adapted directly from the user's robust BaseCamera implementation.
    """
    source_width, source_height = source_wh
    display_width, display_height = display_wh

    assert (
        source_height != 0 and display_height != 0
    ), f"Source height or display height is 0. Source: {source_wh}, Display: {display_wh}"

    source_wh_ratio = source_width / source_height
    display_wh_ratio = display_width / display_height

    if source_wh_ratio > display_wh_ratio:
        # source is "wider" than display, crop the width
        new_width = int(source_height * display_wh_ratio)
        margin = (source_width - new_width) // 2
        cropped_frame = source_frame[:, margin : margin + new_width, :]
    else:
        # source is "taller" than or same as display, crop the height
        new_height = int(source_width / display_wh_ratio)
        margin = (source_height - new_height) // 2
        cropped_frame = source_frame[margin : margin + new_height, :, :]

    return cv2.resize(cropped_frame, display_wh)


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
    align: str = "center",
) -> npt.NDArray[Any]:
    """
    source_frame: (..., H, W, C)
    Crops and resizes a source frame to fit the display resolution while preserving aspect ratio.

    Args:
        align: "left", "center", or "right" for horizontal crop alignment
               (or "top", "center", "bottom" for vertical crop)
    """
    source_height, source_width = source_frame_hwc.shape[-3:-1]
    display_width, display_height = display_wh

    source_wh_ratio = source_width / source_height
    display_wh_ratio = display_width / display_height

    if source_wh_ratio > display_wh_ratio:
        # source is "wider" than display, crop the width
        assert align in ["left", "center", "right"], f"Invalid align: {align}"
        new_width = int(source_height * display_wh_ratio)
        if align == "left":
            margin = 0
        elif align == "right":
            margin = source_width - new_width
        else:  # center
            margin = (source_width - new_width) // 2
        cropped_frame = source_frame_hwc[..., :, margin : margin + new_width, :]
    else:
        # source is "taller" than or same as display, crop the height
        assert align in ["top", "center", "bottom"], f"Invalid align: {align}"
        new_height = int(source_width / display_wh_ratio)
        if align == "top":
            margin = 0
        elif align == "bottom":
            margin = source_height - new_height
        else:  # center
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

