from PIL import Image
import numpy as np
from scipy.stats import skew, kurtosis
from collections.abc import Callable, Iterable, Generator
import math
import argparse
import cv2
import os
from tqdm import tqdm
import csv
from itertools import islice

# TODO: need some ground truth to be able to do:
# TODO: need to train something to determine good feature weights, could do GA

FEATURE_WEIGHTS: tuple[float, ...] = (
    0.08,
    0.16,
    0.08,
    0.06,
    0.12,
    0.06,
    0.02,
    0.04,
    0.02,
    0.01,
    0.02,
    0.01,
    0.08,
    0.16,
    0.08
)

def shot_similarity(shot_1: Iterable[Image.Image], shot_2: Iterable[Image.Image], feature_weights: tuple[float, ...]) -> float:
    # sum of frame matches between shots,
    # divided by the number of frames in the shorter of the shots
    assert sum(feature_weights) > 0.999 and sum(feature_weights) < 1.001, \
        "feature_weights should sum up to 1, but summed to {}".format(sum(feature_weights))

    shot_1_frame_feature_values: list[tuple[float, ...]] = [frame_feature_vector(frame) for frame in shot_1]
    shot_2_frame_feature_values: list[tuple[float, ...]] = [frame_feature_vector(frame) for frame in shot_2]

    min_of_shot_lengths: int = min(len(shot_1_frame_feature_values), len(shot_2_frame_feature_values))

    shot_feature_similarities: list[float] = list()
    for feature_index in range(len(shot_1_frame_feature_values[0])):
        lcs: int = longest_common_subsequence(
            [frame_features[feature_index] for frame_features in shot_1_frame_feature_values],
            [frame_features[feature_index] for frame_features in shot_2_frame_feature_values]
        )
        shot_feature_similarities.append(lcs/min_of_shot_lengths)

    assert len(feature_weights) == len(shot_feature_similarities), \
        "feature_weights should have a len() of {} but had a len() of {}".format(len(shot_feature_similarities), len(feature_weights))

    shot_similarity: float = sum(shot_feature_similarities[i]*feature_weights[i] for i in range(len(feature_weights)))

    return shot_similarity

def frame_feature_vector(frame: Image.Image) -> tuple[float, ...]:
    f1cm: tuple[float, float, float] = frame_first_colour_moment(frame)
    f2cm: tuple[float, float, float] = frame_second_colour_moment(frame)
    f3cm: tuple[float, float, float] = frame_third_colour_moment(frame)
    f4cm: tuple[float, float, float] = frame_fourth_colour_moment(frame)
    ffd: tuple[float, float, float] = frame_fractal_dimension(frame)
    feature_vector: tuple[float, ...] = (
        f1cm[0],
        f1cm[1],
        f1cm[2],
        f2cm[0],
        f2cm[1],
        f2cm[2],
        f3cm[0],
        f3cm[1],
        f3cm[2],
        f4cm[0],
        f4cm[1],
        f4cm[2],
        ffd[0],
        ffd[1],
        ffd[2]
    )
    return feature_vector

def longest_common_subsequence(shot_1_frame_feature_values: list[float], shot_2_frame_feature_values: list[float]) -> int:
    "Each shot is a list of a float-valued feature for each frame in the shot"
    m: int = len(shot_1_frame_feature_values)
    n: int = len(shot_2_frame_feature_values)
    c = np.ndarray((m, n), float)
    for i in range(m):
        c[i, 0] = 0
    for j in range(n):
        c[0, j] = 0
    for i in range(m):
        for j in range(n):
            if shot_1_frame_feature_values[i] == shot_2_frame_feature_values[j]:
                c[i, j] = c[i-1, j-1] + 1
            else:
                c[i, j] = max(c[i, j-1], c[i-1, j])

    return c[m, n]

def frame_first_colour_moment(frame: Image.Image) -> tuple[float, float, float]:
    "Mean / average colour of the frame"
    return frame_colour_moment(frame, np.mean)

def frame_second_colour_moment(frame: Image.Image) -> tuple[float, float, float]:
    "Standard deviation / the colour variance of the frame"
    return frame_colour_moment(frame, np.std)

def frame_third_colour_moment(frame: Image.Image) -> tuple[float, float, float]:
    "Skewness / how asymmetric the colour distribution is in the frame"
    return frame_colour_moment(frame, skew)

def frame_fourth_colour_moment(frame: Image.Image) -> tuple[float, float, float]:
    "Kurtosis / how extreme the tails are compared to normal distribution for the frame"
    return frame_colour_moment(frame, kurtosis)

def frame_colour_moment(frame: Image.Image, fun: Callable) -> tuple[float, float, float]:
    frame = frame.convert("RGB")
    np_image = np.array(frame)
    
    colour_moment: tuple[float, float, float] = (
        float(fun(np_image[:, :, 0])),
        float(fun(np_image[:, :, 1])),
        float(fun(np_image[:, :, 2]))
    )
    return colour_moment

def frame_fractal_dimension(frame: Image.Image) -> tuple[float, float, float]:
    "Fractal dimension index / quantifies fractal complexity of patterns in the frame"
    img = frame.convert('RGB')
    arr = np.array(img)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    r_binarized = r > r.mean()
    g_binarized = g > g.mean()
    b_binarized = b > b.mean()

    fractal_dimension_r: float = fractal_dimension(r_binarized)
    fractal_dimension_g: float = fractal_dimension(g_binarized)
    fractal_dimension_b: float = fractal_dimension(b_binarized)

    return (fractal_dimension_r, fractal_dimension_g, fractal_dimension_b)

def fractal_dimension(Z: np.ndarray) -> float:
    "Generated by ChatGPT"
    assert len(Z.shape) == 2
    Z = Z > 0
    p = min(Z.shape)
    n = 2**int(math.floor(math.log2(p)))

    sizes = 2**np.arange(int(math.log2(n)), 1, -1)
    counts = []

    for size in sizes:
        new_shape = (Z.shape[0] // size * size, Z.shape[1] // size * size)
        Z_crop = Z[:new_shape[0], :new_shape[1]]

        Z_reshaped = Z_crop.reshape(new_shape[0] // size, size, new_shape[1] // size, size)
        Z_blocks = Z_reshaped.any(axis=(1, 3))
        counts.append(np.sum(Z_blocks))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def window_similarity(left_window_shots: list[Iterable[Image.Image]], right_window_shots: list[Iterable[Image.Image]], feature_weights: tuple[float, ...]) -> float:
    window_similarity: float = (
        1 / (len(left_window_shots) * len(right_window_shots))
        * sum(
            sum(
                shot_similarity(left_window_shot, right_window_shot, feature_weights)
                for right_window_shot in right_window_shots
            )
            for left_window_shot in left_window_shots
        )
    )

    return window_similarity

def video_window_similarities(video_path: str, shots_csv_path: str, left_window_size: int, right_window_size: int, feature_weights: tuple[float, ...]) -> Generator[float, None, None]:
    total_shot_count: int = video_total_shot_count(video_path)
    sliding_window_count: int = total_shot_count - left_window_size - right_window_size + 1
    frames: Iterable[Image.Image] = video_to_frames(video_path)
    shot_frame_ranges: Iterable[tuple[int, int]] = video_shot_frame_ranges(shots_csv_path)
    shot_frames: Iterable[Iterable[Image.Image]] = video_shot_frames(frames, shot_frame_ranges)

    left_window_shots: list[Iterable[Image.Image]] = list(islice(shot_frames, left_window_size))
    right_window_shots: list[Iterable[Image.Image]] = list(islice(shot_frames, right_window_size))
    for _ in tqdm(range(sliding_window_count), desc="Computing Window Similarities", total=sliding_window_count):
        left_window_shots.pop(0)
        left_window_shots.append(right_window_shots.pop(0))
        right_window_shots.append(next(shot_frames))

        yield window_similarity(left_window_shots, right_window_shots, feature_weights)

def video_to_frames(video_path: str) -> Generator[Image.Image, None, None]:
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image: Image.Image = Image.fromarray(frame_rgb)
        
        yield frame_image  # Yield one frame at a time

    video.release()

def video_total_frame_count(video_path: str) -> int:
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return total_frames

def video_shot_frame_ranges(shots_csv_path: str) -> Generator[tuple[int, int], None, None]:
    with open(shots_csv_path, "r") as shots_csv:
        shots_csv_reader = csv.reader(shots_csv)
        for row in shots_csv_reader:
            if row[0].isnumeric() and row[1].isnumeric() and len(row) == 4:
                # skip header / last / invalid rows
                yield (int(row[0]), int(row[1]))

def video_total_shot_count(shots_csv_path: str) -> int:
    shot_count: int = 0
    with open(shots_csv_path, "r") as shots_csv:
        shots_csv_reader = csv.reader(shots_csv)
        for row in shots_csv_reader:
            shot_count += int(row[0].isnumeric() and row[1].isnumeric() and len(row) == 4)

    return shot_count

def video_shot_frames(frames: Iterable[Image.Image], shot_frame_ranges: Iterable[tuple[int, int]]) -> Generator[Iterable[Image.Image], None, None]:
    last_shot_frame_end: int = -1
    for shot_frame_start, shot_frame_end in shot_frame_ranges:
        assert shot_frame_start == last_shot_frame_end + 1
        shot_frame_count: int = shot_frame_end - shot_frame_start + 1
        shot_frames: Iterable[Image.Image] = islice(frames, shot_frame_count)

        yield shot_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot Generator Using Cosine Similarity")
    parser.add_argument('--video', required=True, type=str, help="Path to Video File")
    parser.add_argument('--shots', required=True, type=str, help="Path to CSV Containing Shots")
    parser.add_argument('--left-window-size', type=int, default=3, help="Number of Shots in Left Sliding Window")
    parser.add_argument('--right-window-size', type=int, default=3, help="Number of Shots in Right Sliding Window")
    args = parser.parse_args()
    video_path: str = args.video
    assert os.path.exists(video_path) and os.path.isfile(video_path), "Could not find a file at the provided video path"
    shots_csv_path: str = args.shots
    assert os.path.exists(shots_csv_path) and os.path.isfile(shots_csv_path), "Could not find a file at the provided shots CSV path"
    left_window_size: int = args.left_window_size
    assert left_window_size > 0, "left window size must be greater than 0"
    right_window_size: int = args.right_window_size
    assert right_window_size > 0, "right window size must be greater than 0"
