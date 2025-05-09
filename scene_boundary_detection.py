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
    0.10,
    0.16,
    0.10,
    0.08,
    0.16,
    0.08,
    0.08,
    0.16,
    0.08
)

def shot_similarity(shot_1: Iterable[Image.Image], shot_2: Iterable[Image.Image], feature_weights: tuple[float, ...]) -> float:
    # sum of frame matches between shots,
    # divided by the number of frames in the shorter of the shots
    assert sum(feature_weights) > 0.999 and sum(feature_weights) < 1.001, \
        "feature_weights should sum up to 1, but summed to {}".format(sum(feature_weights))

    shot_1_frame_feature_values: list[tuple[bool, ...]] = [tuple([binarize_float(value) for value in frame_feature_vector(frame)]) for frame in shot_1]
    shot_2_frame_feature_values: list[tuple[bool, ...]] = [tuple([binarize_float(value) for value in frame_feature_vector(frame)]) for frame in shot_2]

    min_of_shot_lengths: int = min(len(shot_1_frame_feature_values), len(shot_2_frame_feature_values))
    assert min_of_shot_lengths > 0, "shots must have more than 0 frames. shot_1: {} frames, shot_2: {} frames".format(len(shot_1_frame_feature_values), len(shot_2_frame_feature_values))

    shot_feature_similarities: list[float] = list()
    for feature_index in range(len(shot_1_frame_feature_values[0])):
        lcs: int = longest_common_subsequence_length(
            [frame_features[feature_index] for frame_features in shot_1_frame_feature_values],
            [frame_features[feature_index] for frame_features in shot_2_frame_feature_values]
        )
        shot_feature_similarities.append(lcs/min_of_shot_lengths)

    assert len(feature_weights) == len(shot_feature_similarities), \
        "feature_weights should have a len() of {} but had a len() of {}".format(len(shot_feature_similarities), len(feature_weights))

    shot_similarity: float = sum(shot_feature_similarities[i]*feature_weights[i] for i in range(len(feature_weights)))

    assert shot_similarity >= 0
    assert shot_similarity <= 1

    return shot_similarity

def frame_feature_vector(frame: Image.Image) -> tuple[float, ...]:
    f1cm: tuple[float, float, float] = frame_first_colour_moment(frame)
    f2cm: tuple[float, float, float] = frame_second_colour_moment(frame)
    ffd: tuple[float, float, float] = frame_fractal_dimension(frame)
    feature_vector: tuple[float, ...] = (
        f1cm[0],
        f1cm[1],
        f1cm[2],
        f2cm[0],
        f2cm[1],
        f2cm[2],
        ffd[0],
        ffd[1],
        ffd[2]
    )
    return feature_vector

def binarize_float(value: float, least: float = 0.0, greatest: float = 1.0) -> bool:
    assert value >= least, "value of {} was not greater than least of {}".format(value, least)
    assert value <= greatest, "value of {} was not less than greatest of {}".format(value, greatest)
    # the above assertions also imply that greatest >= least
    return (value >= (least + (greatest - least)/2))

def longest_common_subsequence_length(sequence_1: list[int], sequence_2: list[int]) -> int:
    "Each shot is a list of a float-valued feature for each frame in the shot"
    m: int = len(sequence_1)
    n: int = len(sequence_2)
    c = np.zeros((m+1, n+1), int)
    for i in range(m):
        for j in range(n):
            if sequence_1[i] == sequence_2[j]:
                c[i+1, j+1] = c[i, j] + 1
            else:
                c[i+1, j+1] = max(c[i+1, j], c[i, j+1])

    return c[m, n]

def frame_first_colour_moment(frame: Image.Image) -> tuple[float, float, float]:
    "Mean / average colour of the frame"
    return frame_colour_moment(frame, np.mean)

def frame_second_colour_moment(frame: Image.Image) -> tuple[float, float, float]:
    "Standard deviation / the colour variance of the frame"
    return frame_colour_moment(frame, np.std)

def frame_colour_moment(frame: Image.Image, fun: Callable) -> tuple[float, float, float]:
    frame = frame.convert("RGB")
    np_image = np.array(frame)

    try:
        colour_moment: tuple[float, float, float] = (
            float(fun(np_image[:, :, 0]))/255,
            float(fun(np_image[:, :, 1]))/255,
            float(fun(np_image[:, :, 2]))/255
        )
    except TypeError:
        colour_moment: tuple[float, float, float] = (0, 0, 0)

    for moment in colour_moment:
        assert not math.isnan(moment)
        assert moment >= 0, "invalid colour moment of {}".format(moment)
        assert moment <= 1, "invalid colour moment of {}".format(moment)

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
    if not np.any(Z):
        return 0.0 # no structure

    p = min(Z.shape)
    n = 2**int(math.floor(math.log2(p)))

    sizes = 2**np.arange(int(math.log2(n)), 1, -1)
    counts = []

    for size in sizes:
        new_shape = (Z.shape[0] // size * size, Z.shape[1] // size * size)
        Z_crop = Z[:new_shape[0], :new_shape[1]]

        if Z_crop.size == 0:
            continue

        Z_reshaped = Z_crop.reshape(new_shape[0] // size, size, new_shape[1] // size, size)
        Z_blocks = Z_reshaped.any(axis=(1, 3))
        count = np.sum(Z_blocks)
        if count == 0:
            continue
        counts.append(count)

    if len(counts) < 2:
        return 0.0 # not enough data to fit a polynomial

    coeffs = np.polynomial.polynomial.Polynomial.fit(np.log(sizes), np.log(counts), 1).convert().coef

    fractal_dimension = -coeffs[0] if not math.isnan(-coeffs[0]) else 0.0

    assert not math.isnan(fractal_dimension)

    fractal_dimension /= 2 # go from [0, 2] to [0, 1]

    if fractal_dimension < 0 or fractal_dimension > 1:
        return 0.0 # invalid fractal dimension, return default valid

    return fractal_dimension

def window_similarity(left_window_shots: list[list[Image.Image]], right_window_shots: list[list[Image.Image]], feature_weights: tuple[float, ...]) -> float:
    shot_similarity_sum: float = 0
    for left_i in range(len(left_window_shots)):
        for right_i in range(len(right_window_shots)):
            shot_similarity_sum += shot_similarity(left_window_shots[left_i], right_window_shots[right_i], feature_weights)

    window_similarity: float = shot_similarity_sum / (len(left_window_shots) * len(right_window_shots))

    return window_similarity

def video_window_similarities(video_path: str, shots_csv_path: str, left_window_size: int, right_window_size: int, feature_weights: tuple[float, ...]) -> Generator[float, None, None]:
    total_shot_count: int = video_total_shot_count(shots_csv_path)
    sliding_window_count: int = total_shot_count - left_window_size - right_window_size + 1
    frames: Iterable[Image.Image] = video_to_frames(video_path)
    shot_frame_ranges: Iterable[tuple[int, int]] = video_shot_frame_ranges(shots_csv_path)
    shot_frames: Iterable[list[Image.Image]] = video_shot_frames(frames, shot_frame_ranges)

    left_window_shots: list[list[Image.Image]] = list(islice(shot_frames, left_window_size))
    assert len(left_window_shots) > 0
    right_window_shots: list[list[Image.Image]] = list(islice(shot_frames, right_window_size))
    assert len(right_window_shots) > 0
    yield window_similarity(left_window_shots, right_window_shots, feature_weights)
    for i in tqdm(range(sliding_window_count), desc="Computing Window Similarities", total=sliding_window_count):
        left_window_shots.pop(0)
        left_window_shots.append(right_window_shots.pop(0))
        right_window_shots.append(next(shot_frames))
        assert len(left_window_shots) > 0
        assert len(right_window_shots) > 0

        yield window_similarity(left_window_shots, right_window_shots, feature_weights)

def video_to_frames(video_path: str) -> Generator[Image.Image, None, None]:
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Extracting Frames from Video", total=total_frames):
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
        for row in tqdm(shots_csv_reader, desc="Reading Shot Frame Ranges"):
            if row[0].isnumeric() and row[1].isnumeric() and len(row) == 4:
                # skip header / last / invalid rows
                yield (int(row[0]), int(row[1]))

def video_total_shot_count(shots_csv_path: str) -> int:
    shot_count: int = 0
    with open(shots_csv_path, "r") as shots_csv:
        shots_csv_reader = csv.reader(shots_csv)
        for row in tqdm(shots_csv_reader, desc="Counting Number of Shots"):
            shot_count += int(row[0].isnumeric() and row[1].isnumeric() and len(row) == 4)

    return shot_count

def video_shot_frames(frames: Iterable[Image.Image], shot_frame_ranges: Iterable[tuple[int, int]]) -> Generator[list[Image.Image], None, None]:
    last_shot_frame_end: int = -1
    for shot_frame_start, shot_frame_end in tqdm(shot_frame_ranges, desc="Grouping Frames into Shots"):
        assert shot_frame_start == last_shot_frame_end + 1
        shot_frame_count: int = shot_frame_end - shot_frame_start + 1
        assert shot_frame_count > 0
        shot_frames: list[Image.Image] = list(islice(frames, shot_frame_count))
        last_shot_frame_end = shot_frame_end

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

    for video_window_similarity in video_window_similarities(video_path, shots_csv_path, left_window_size, right_window_size, FEATURE_WEIGHTS):
        print(video_window_similarity)

# TODO: some calculations are performed multiple times on the same information, optimize to avoid recalculating
