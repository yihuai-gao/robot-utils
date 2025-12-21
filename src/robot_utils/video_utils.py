from typing import Any
import numpy as np
import numpy.typing as npt
import imageio

def save_np_array_as_video(
    rollout_images: list[npt.NDArray[Any]] | npt.NDArray[np.uint8],
    video_path: str,
    fps: int = 30,
):
    """Saves an MP4 replay of an episode."""

    video_writer = imageio.get_writer(video_path, fps=fps)
    for img in rollout_images:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {video_path}")

