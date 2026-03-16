from typing import Any
import numpy as np
import numpy.typing as npt
import imageio

import glob
import os
import cv2
import imageio
from moviepy import (
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from moviepy import VideoFileClip, clips_array, vfx





def save_np_array_as_video(rollout_images: Any, video_path: str, fps: int = 10):
    """
    rollout_images: (T, H, W, C) or (T, C, H, W)
    Saves an MP4 replay of an episode.
    """

    video_writer = imageio.get_writer(video_path, fps=fps)
    for img in rollout_images:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {video_path}")

def merge_videos(
    video_paths: list[str],
    success_labels: list[bool],
    output_path: str,
    rows: int,
    cols: int,
    margin_size: int,
    extend_time: float = 1,
    blend_alpha: float = 0.2,
    cropping_top_down_left_right: tuple[float, float, float, float] = (0, 0, 0, 0),
    max_time: float = -1,
    speed: float = 1,
    vertical_first: bool = True,
    show_speed_label: bool = True,
):
    os.environ["FFMPEG_BINARY"] = f"{os.environ['CONDA_PREFIX']}/bin/ffmpeg"

    # --- Configuration ---
    # 1. Path to your video folder
    # 2. Output file name
    expected_clips = rows * cols
    encoder = "h264"
    # encoder = "h264"
    ffmpeg_params = ["-c:v", encoder]

    assert cropping_top_down_left_right[0] + cropping_top_down_left_right[1] < 1
    assert cropping_top_down_left_right[2] + cropping_top_down_left_right[3] < 1

    assert len(video_paths) == expected_clips
    assert len(success_labels) == expected_clips

    # 2. Load clips and find the maximum duration
    video_clips: list[VideoFileClip] = []
    max_duration = 0
    print("Loading clips and determining maximum duration...")

    for i, path in enumerate(video_paths):
        try:
            clip = VideoFileClip(path)
            if max_time > 0:
                clip = clip.subclipped(0, max_time)
            clip = clip.resized(new_size=1 / max(rows, cols))
            clip = clip.with_speed_scaled(speed)
            assert isinstance(clip, VideoFileClip)
            clip_width, clip_height = clip.w, clip.h
            cropping = vfx.Crop(
                y1=cropping_top_down_left_right[0] * clip_height,
                y2=(1 - cropping_top_down_left_right[1]) * clip_height,
                x1=cropping_top_down_left_right[2] * clip_width,
                x2=(1 - cropping_top_down_left_right[3]) * clip_width,
            )
            clip = cropping.apply(clip)
            add_margin = vfx.Margin(
                color=(0, 0, 0),
                margin_size=margin_size,
            )
            clip = add_margin.apply(clip)
            assert isinstance(clip, VideoFileClip)
            video_clips.append(clip)
            if isinstance(clip.duration, float) and clip.duration > max_duration:
                max_duration = clip.duration
        except Exception as e:
            print(f"Could not load clip {path}: {e}")
            return

    print(
        f"Maximum video duration found: {max_duration:.2f} seconds. Will extend to {max_duration + extend_time: .2f} seconds"
    )
    final_duration = max_duration + extend_time

    # 3. Standardize duration (Keep last frame if shorter)
    extended_clips = []
    for i, clip in enumerate(video_clips):
        # Overlay a green or red transparent mask on the last frame of the clip
        if success_labels[i]:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Red
        assert isinstance(clip.duration, float)

        last_frame = clip.get_frame(clip.duration - 1 / float(clip.fps))
        assert isinstance(last_frame, np.ndarray)
        last_frame = (
            last_frame * (1 - blend_alpha)
            + np.ones_like(last_frame) * np.array(color)[None, None, :] * blend_alpha
        )
        last_frame = last_frame.astype(np.uint8)

        additional_duration = final_duration - clip.duration
        last_frame_clip = ImageClip(last_frame).with_duration(additional_duration)

        extended_clip = concatenate_videoclips([clip, last_frame_clip])
        extended_clips.append(extended_clip)
        print(
            f"  - Extended clip {i+1} from {clip.duration:.2f}s to {final_duration:.2f}s (Last frame freeze)."
        )

    # 4. Arrange clips into a 4x5 grid structure
    # The clips_array function takes a 2D list (rows of columns)
    try:

        if vertical_first:
            # Column-first order: fills down each column before moving right
            clip_cols = []
            for i in range(cols):
                start_index = i * rows
                end_index = start_index + rows
                clip_cols.append(extended_clips[start_index:end_index])
            clip_rows = list(zip(*clip_cols))
        else:
            # Row-first order: fills across each row before moving down
            clip_rows = []
            for i in range(rows):
                start_index = i * cols
                end_index = start_index + cols
                clip_rows.append(extended_clips[start_index:end_index])

        final_clip = clips_array(clip_rows)

    except Exception as e:
        print(f"Error during clips_array arrangement: {e}")
        return

    print(f"\nExporting final video to '{output_path}' using {encoder}...")

    if show_speed_label:
        overlay_text = TextClip(
            font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            text=f"Speed:{speed}x",
            color="white",
            font_size=30,
            size=(1000, 100),
            margin=(10, 10),
            method="caption",
            text_align="left",
            vertical_align="left",
            horizontal_align="top",
            duration=final_clip.duration,
        )
        final_clip = CompositeVideoClip([final_clip, overlay_text])

    final_clip.write_videofile(
        output_path,
        codec=encoder,
        # audio_codec=audio_codec,
        # temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        # CRITICAL: This passes the GPU encoder flag to FFmpeg
        ffmpeg_params=ffmpeg_params,
        # # Set to 1 for faster, single-threaded processing; increase if needed
        # n_jobs=1,
        # Set a high bitrate or a quality factor (CRF) for better quality
        bitrate="5000k",  # 5 Mbps
    )
    print(f"\n--- SUCCESS! Video saved to {output_path} ---")
