from PIL import Image
import numpy as np
from collections.abc import Callable, Iterable, Generator
import math
import argparse
import cv2
import os
from tqdm import tqdm
import csv
from itertools import islice
from typing import TypeVar, Optional
import matplotlib.pyplot as plt
import multiprocessing

from FractalDimension import FractalDimension

# TODO: use ground truth to find:
# best quantization levels with some reasonable feature weights, then
# best feature weights with that quantization levels

FeatureVector = tuple[float, float, float, float, float, float, float, float, float]
FeatureWeights = tuple[float, float, float, float, float, float, float, float, float]

FEATURE_WEIGHTS: FeatureWeights = (
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

FEATURE_QUANTIZATION_LEVELS: int = 16

LOCAL_MEAN_SMOOTHING_WINDOW_SIZE: int = 3

PLOT_FIGURE_SIZE_INCHES: tuple[float, float] = (36.0, 18.0)
PLOT_FIGURE_DPI: int = 200
PLOT_HISTOGRAM_NUMBER_OF_BINS: int = 20

FRAME_FEATURE_BUFFER_SIZE: int = 10000

T = TypeVar("T")

FRAME_EXTRACTION_LIMIT: int | None = None # to shorten total running time for testing purposes

def shot_similarity(shot_1: Iterable[FeatureVector], shot_2: Iterable[FeatureVector], feature_weights: FeatureWeights, feature_quantization_levels: int) -> float:
    # sum of frame matches between shots,
    # divided by the number of frames in the shorter of the shots
    assert sum(feature_weights) > 0.999 and sum(feature_weights) < 1.001, \
        "feature_weights should sum up to 1, but summed to {}".format(sum(feature_weights))

    if shot_1 == shot_2:
        # if they are literally the same shot with the same frames (the same object), then obviously their similarity is 1.0
        print("WARNING: used shot_similarity function to compare identical shot objects, this is likely not intended behaviour")
        return 1.0

    shot_1_frame_feature_values: list[tuple[int, ...]] = [tuple([quantize_float(value, feature_quantization_levels) for value in frame]) for frame in shot_1]
    shot_2_frame_feature_values: list[tuple[int, ...]] = [tuple([quantize_float(value, feature_quantization_levels) for value in frame]) for frame in shot_2]

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

def frame_feature_vector(frame: Image.Image) -> FeatureVector:
    f1cm: tuple[float, float, float] = frame_first_colour_moment(frame)
    f2cm: tuple[float, float, float] = frame_second_colour_moment(frame)
    ffd: tuple[float, float, float] = frame_fractal_dimension(frame)
    feature_vector: FeatureVector = (
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

def quantize_float(value: float, quantization_levels: int, least: float = 0.0, greatest: float = 1.0) -> int:
    assert quantization_levels >= 2, "quantization_levels must be greater than or equal to 2"
    assert value >= least, "value of {} was not greater than least of {}".format(value, least)
    assert value <= greatest, "value of {} was not less than greatest of {}".format(value, greatest)
    # the above assertions also imply that greatest >= least
    normalized_value: float = (value - least) / (greatest - least) # normalized to [0, 1] range
    quantization_level: int = int(normalized_value * quantization_levels)
    quantization_level = min(max(quantization_level, 0), quantization_levels - 1) # clamp to ensure valid output

    return quantization_level

def longest_common_subsequence_length(sequence_1: list[T], sequence_2: list[T]) -> int:
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

    fractal_dimension_r: float = fractal_dimension(r)
    fractal_dimension_g: float = fractal_dimension(g)
    fractal_dimension_b: float = fractal_dimension(b)

    return (fractal_dimension_r, fractal_dimension_g, fractal_dimension_b)

def fractal_dimension(grayscale_image: np.ndarray) -> float:
    target_shape: tuple[int, int] = (max(grayscale_image.shape), max(grayscale_image.shape))
    stretched_image: np.ndarray = cv2.resize(grayscale_image, target_shape, interpolation=cv2.INTER_LINEAR)
    return float(FractalDimension.fractal_dimension(stretched_image))

def window_similarity(left_window_shots: list[list[FeatureVector]], right_window_shots: list[list[FeatureVector]], feature_weights: FeatureWeights, feature_quantization_levels: int) -> float:
    if left_window_shots == right_window_shots:
        # if they are literally the same window with the same shots (the same object), then obviously their similarity is 1.0
        print("WARNING: used window_similarity function to compare identical window objects, this is likely not intended behaviour")
        return 1.0

    shot_similarity_sum: float = 0.0
    for left_window_shot in left_window_shots:
        for right_window_shot in right_window_shots:
            shot_similarity_sum += shot_similarity(left_window_shot, right_window_shot, feature_weights, feature_quantization_levels)

    window_similarity: float = shot_similarity_sum / (len(left_window_shots) * len(right_window_shots))

    assert window_similarity >= 0.0
    assert window_similarity <= 1.0

    return window_similarity

def video_window_similarities(frame_feature_vectors_queue: "multiprocessing.Queue[Optional[FeatureVector]]", results_queue: "multiprocessing.Queue[Optional[float]]", shots_csv_path: str, left_window_size: int, right_window_size: int, feature_weights: FeatureWeights, feature_quantization_levels: int) -> None:
    assert left_window_size > 0
    assert right_window_size > 0
    total_shot_count: int = video_total_shot_count(shots_csv_path)
    sliding_window_count: int = total_shot_count - left_window_size - right_window_size + 1
    frames: Iterable[FeatureVector] = queue_to_iterator(frame_feature_vectors_queue)
    shot_frames: Iterable[list[FeatureVector]] = video_shot_frames(frames, shots_csv_path)

    left_window_shots: list[list[FeatureVector]] = list(islice(shot_frames, left_window_size))
    assert len(left_window_shots) == left_window_size
    right_window_shots: list[list[FeatureVector]] = list(islice(shot_frames, right_window_size))
    assert len(right_window_shots) == right_window_size
    assert left_window_shots != right_window_shots
    results_queue.put(window_similarity(left_window_shots, right_window_shots, feature_weights, feature_quantization_levels))
    for shot_frame in tqdm(shot_frames, desc="Computing Window Similarities", total=sliding_window_count):
        left_window_shots.pop(0)
        left_window_shots.append(right_window_shots.pop(0))
        right_window_shots.append(shot_frame)
        assert len(left_window_shots) == left_window_size
        assert len(right_window_shots) == right_window_size
        assert left_window_shots != right_window_shots

        try:
            results_queue.put(window_similarity(left_window_shots, right_window_shots, feature_weights, feature_quantization_levels))
        except Exception as e:
            print(e)
            results_queue.put(None)
            return

    results_queue.put(None)

def threaded_video_window_similarities(results_queue: "multiprocessing.Queue[Optional[float]]", video_path: str, shots_csv_path: str, left_window_size: int, right_window_size: int, feature_weights: FeatureWeights, feature_quantization_levels: int) -> None:
    assert left_window_size > 0
    assert right_window_size > 0

    frame_feature_vectors_queue: "multiprocessing.Queue[Optional[tuple[float, ...]]]" = multiprocessing.Queue(maxsize=FRAME_FEATURE_BUFFER_SIZE)
    producer_process: multiprocessing.Process = multiprocessing.Process(target=video_to_frame_feature_vectors, args=(video_path, frame_feature_vectors_queue))
    consumer_process: multiprocessing.Process = multiprocessing.Process(target=video_window_similarities, args=(frame_feature_vectors_queue, results_queue, shots_csv_path, left_window_size, right_window_size, feature_weights, feature_quantization_levels))

    producer_process.start()
    consumer_process.start()
    producer_process.join()
    consumer_process.join()    

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

def video_to_frame_feature_vectors(video_path: str, frame_feature_vectors_queue: "multiprocessing.Queue[Optional[tuple[float, ...]]]") -> None:
    # uses a cache to skip loading frames and calculating features if it has already been done once
    total_frame_count: int = video_total_frame_count(video_path)
    video_name: str = os.path.splitext(os.path.basename(video_path))[0]
    frame_feature_cache_path: str = f"./frame_feature_cache/{video_name}.csv"
    os.makedirs(os.path.dirname(frame_feature_cache_path), exist_ok=True)
    header: tuple[str, ...] = (
        "Frame Index",
        "First Colour Moment (R)",
        "First Colour Moment (G)",
        "First Colour Moment (B)",
        "Second Colour Moment (R)",
        "Second Colour Moment (G)",
        "Second Colour Moment (B)",
        "Fractal Dimension (R)",
        "Fractal Dimension (G)",
        "Fractal Dimension (B)"
    )
    try:
        with open(frame_feature_cache_path, "r") as cache_file:
            cache_reader = csv.reader(cache_file)
            frame_counter: int = 0
            feature_vectors: list[FeatureVector] = list()
            for i, row in tqdm(enumerate(cache_reader), desc="Reading Cached Frame Features", total=total_frame_count):
                if i == 0:
                    assert tuple(row) == header # if the header is different, cache is likely invalid
                    continue
                assert len(row) == len(header)
                frame_index: int = int(row[0])
                assert frame_index == i-1
                feature_vector: FeatureVector = (
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9])
                )
                assert len(row) == len(feature_vector)+1
                feature_vectors.append(feature_vector) # cannot push to queue until we are sure the cache is valid
                frame_counter += 1

            assert frame_counter == total_frame_count # ensures that cache wasn't missing any frames
            # cache is assumed to have been valid at this point, push all feature vectors to queue
            [frame_feature_vectors_queue.put(feature_vector) for feature_vector in feature_vectors]
    except:
        # failed to read frame features from cache (whether because cache does not exist yet, or some other reason)
        # clear cache for this video by opening for writing, and write cache
        print("Frame feature cache invalid or non-existant, reading frames, computing features, and caching...")
        with open(frame_feature_cache_path, "w", newline="") as cache_file:
            cache_writer = csv.writer(cache_file)
            cache_writer.writerow(header)
            for i, frame in tqdm(enumerate(video_to_frames(video_path)), desc="Reading Frames and Computing Features", total=total_frame_count):
                feature_vector: FeatureVector = frame_feature_vector(frame)
                frame_feature_vectors_queue.put(feature_vector)
                csv_row: list = [i]
                csv_row.extend(feature_vector)
                assert len(csv_row) == len(header)
                cache_writer.writerow(csv_row)
                cache_file.flush()
                if FRAME_EXTRACTION_LIMIT is not None and i == FRAME_EXTRACTION_LIMIT:
                    break

    frame_feature_vectors_queue.put(None)

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

def video_shot_frames(frames: Iterable[T], shots_csv_path: str) -> Generator[list[T], None, None]:
    shot_frame_ranges: Iterable[tuple[int, int]] = video_shot_frame_ranges(shots_csv_path)
    total_shot_count: int = video_total_shot_count(shots_csv_path)
    last_shot_frame_end: int = -1
    for shot_frame_start, shot_frame_end in tqdm(shot_frame_ranges, desc="Grouping Frames into Shots", total=total_shot_count):
        assert shot_frame_start == last_shot_frame_end + 1
        shot_frame_count: int = shot_frame_end - shot_frame_start + 1
        assert shot_frame_count > 0
        shot_frames: list[T] = list(islice(frames, shot_frame_count))
        last_shot_frame_end = shot_frame_end

        yield shot_frames

def queue_to_iterator(q: "multiprocessing.Queue[Optional[T]]") -> Generator[T, None, None]:
    while True:
        item = q.get()
        if item is None:
            break
        yield item

def local_mean_smoothing(data: list[float], window_size: int) -> list[float]:
    assert isinstance(window_size, int)
    assert window_size > 0
    if window_size == 1:
        return data
    kernel = np.ones(window_size) / window_size
    smoothed: list[float] = list(np.convolve(data, kernel, mode='same'))

    return smoothed

def generate_scene_boundary_indices(window_similarities: Iterable[float], window_similarity_count: int, left_window_size: int, right_window_size: int, local_window_threshold: float) -> Generator[int, None, None]:
    assert window_similarity_count > 0
    assert left_window_size > 0
    assert right_window_size > 0
    assert local_window_threshold > 0
    assert local_window_threshold < 1

    largest_window: int = max(left_window_size, right_window_size)

    last_window_similarity: Optional[float] = None
    last_boundary_i: int = 0
    for i, window_similarity in tqdm(enumerate(window_similarities), desc="Comparing Window Similarities to Threshold", total=window_similarity_count):
        assert window_similarity >= 0
        assert window_similarity <= 1
        assert i < window_similarity_count

        if last_window_similarity is None:
            last_window_similarity = window_similarity
            continue

        if (i - last_boundary_i) < largest_window:
            # not enough shots to be able to detect a scene change
            last_window_similarity = window_similarity
            continue

        if abs(window_similarity - last_window_similarity) > local_window_threshold:
            last_boundary_i = i
            yield i

        last_window_similarity = window_similarity

    yield window_similarity_count-1


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

    window_similarities_queue: "multiprocessing.Queue[Optional[float]]" = multiprocessing.Queue()
    consumer_process = multiprocessing.Process(target=threaded_video_window_similarities, args=(window_similarities_queue, video_path, shots_csv_path, left_window_size, right_window_size, FEATURE_WEIGHTS, FEATURE_QUANTIZATION_LEVELS))
    consumer_process.start()

    window_similarities: list[float] = list()
    with open("video_window_similarities.txt", "w") as file:
        for video_window_similarity in queue_to_iterator(window_similarities_queue):
            file.write(f"{video_window_similarity}\n")
            file.flush()
            window_similarities.append(video_window_similarity)

    consumer_process.join()

    smoothed_window_similarities: list[float] = local_mean_smoothing(window_similarities, LOCAL_MEAN_SMOOTHING_WINDOW_SIZE)

    plt.figure(figsize=PLOT_FIGURE_SIZE_INCHES)
    plt.plot(window_similarities)
    plt.plot(smoothed_window_similarities)
    plt.legend(("Window Similarities", "Smoothed Window Similarities"))
    plt.xticks(np.arange(0, len(window_similarities)+0.1, 30), minor = False)
    plt.xticks(np.arange(0, len(window_similarities)+0.1, 10), minor = True)
    plt.yticks(np.linspace(0, 1, 5), minor = False)
    plt.yticks(np.linspace(0, 1, 10), minor = True)
    plt.grid(True, "major", "y", linewidth = 2, alpha = 0.5)
    plt.grid(True, "minor", "y", linewidth = 2, alpha = 0.1)
    plt.grid(True, "major", "x", linewidth = 2, alpha = 0.5)
    plt.grid(True, "minor", "x", linewidth = 2, alpha = 0.1)
    plt.xlabel("Shot #")
    plt.ylabel("Window Similarity (0-1)")
    plt.title("Window Similarity Over the Video")
    plt.savefig("window_similarities_line_plot.webp", format="webp", pil_kwargs={'lossless': True}, dpi=PLOT_FIGURE_DPI)

    plt.figure(figsize=PLOT_FIGURE_SIZE_INCHES)
    plt.hist(window_similarities, bins=list(np.linspace(0, 1, PLOT_HISTOGRAM_NUMBER_OF_BINS+1)), edgecolor="black")
    plt.title("Histogram of Window Similarities")
    plt.xlabel("Window Similarity (0-1)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("window_similarities_histogram.webp", format="webp", pil_kwargs={'lossless': True}, dpi=PLOT_FIGURE_DPI)

# TODO: some additional efficiency could still be obtained, since frame features are compared (with LCS) multiple times as windows of size >1 slide

# TODO: ability to run script with multiple quantization levels / feature weights in parallel

# TODO: could also cache LCS results between two frames
