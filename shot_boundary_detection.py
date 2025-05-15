import csv
from tqdm import tqdm
import os
from collections.abc import Iterable, Generator
import json
from ffmpeg import FFmpeg
import pandas as pd
import argparse

SHOT_BOUNDARY_CSV_DIRECTORY: str = "./shot_boundaries/"

def generate_shot_boundary_indices(total_frame_count: int, cosine_similarities: Iterable[tuple[int, int, float]], threshold: float) -> Generator[int, None, None]:
    for i, _, similarities in tqdm(cosine_similarities, desc="Comparing Frame Similarities to Threshold", total=total_frame_count):
        if similarities < threshold:
            yield i

def shot_boundary_indices_to_tuples(shot_boundary_indices: Iterable[int]) -> Generator[tuple[int, int], None, None]:
    last_shot_boundary_index: int = 0
    for shot_boundary_index in shot_boundary_indices:
        entry: tuple[int, int] = (last_shot_boundary_index, shot_boundary_index)
        yield entry

        last_shot_boundary_index = shot_boundary_index + 1

def write_shot_boundary_tuples_to_csv(shot_boundary_tuples: Iterable[tuple[int, int]], output_path: str) -> None:
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Start Frame", "End Frame"])
        for row in shot_boundary_tuples:
            assert len(row) == 2, "shot boundary tuple should have contained 2 items"
            writer.writerow(row)
            file.flush()

def convert_seconds(seconds: float | int) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds: float | int = seconds % 60  # Keep decimal precision if needed

    return f"{hours:02}:{minutes:02}:{remaining_seconds:06.3f}"

def convert_frame_file_to_seconds(shots_file_csv,video_path,output_path):
    probe = json.loads(FFmpeg(executable="ffprobe").input(video_path, print_format="json", show_streams=None).execute())
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    frame_rate = eval(video_streams[0]['r_frame_rate'])
    df = pd.read_csv(shots_file_csv)
    df["Start Time"] = df["Start Frame"].apply(lambda x: x/frame_rate)
    df["Start Time"] = df["Start Time"].apply(convert_seconds)
    df["End Time"] = df["End Frame"].apply(lambda x: x/frame_rate)
    df["End Time"] = df["End Time"].apply(convert_seconds)
    df.to_csv(output_path,index=False)

def video_total_frame_count(cosine_similarities_csv_path: str) -> int:
    total_frames: int = 0
    with open(cosine_similarities_csv_path, "r") as file:
        similarities_reader = csv.reader(file)
        for i, row in tqdm(enumerate(similarities_reader), desc="Counting Number of Frames"):
            total_frames += 1
            # also validate that the CSV is in the expected format
            assert len(row) == 3
            if i > 0:
                assert int(row[1])+1 == total_frames, f"{row[1]}, {total_frames}"

    return total_frames

def video_cosine_similarities(cosine_similarities_csv_path: str, total_frames: int) -> list[tuple[int, int, float]]:
    cosine_similarities: list[tuple[int, int, float]] = list()
    with open(cosine_similarities_csv_path, "r") as file:
        similarities_reader = csv.reader(file)
        for i, row in tqdm(enumerate(similarities_reader), desc="Fetching Precomputed Cosine Similarities", total=total_frames):
            assert len(row) == 3
            if i == 0:
                continue # skip header
            try:
                cosine_similarities.append((int(row[0]), int(row[1]), float(row[2])))
            except:
                print(f"Cosine Similarities CSV row {i} invalid format")

    return cosine_similarities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot Generator Using Cosine Similarity")
    parser.add_argument('--cosine-similarities-csv-path', "-csv", required=True, type=str, help="Path to CSV file containing the cosine similarities")
    parser.add_argument('--video-path', "--video", required=True, type=str, help="Path to video file")
    parser.add_argument('--threshold', required=True, type=float, nargs="*", help="One of more threshold values for cosine similarities (between 0 and 1)")
    args = parser.parse_args()
    cosine_similarities_csv_path: str = args.cosine_similarities_csv_path
    video_path: str = args.video_path
    thresholds: list[float] = args.threshold
    assert isinstance(thresholds, list)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frame_count: int = video_total_frame_count(cosine_similarities_csv_path)
    cosine_similarities: list[tuple[int, int, float]] = video_cosine_similarities(cosine_similarities_csv_path, total_frame_count)
    for threshold in thresholds:
        assert isinstance(threshold, float)
        assert threshold > 0, "Thresholds must be greater than 0"
        assert threshold < 1, "Thresholds must be less than 1"

        shot_boundaries_csv_path: str = os.path.join(SHOT_BOUNDARY_CSV_DIRECTORY, base_name, f"{threshold:.3f}_shot_boundaries.csv")
        os.makedirs(os.path.dirname(shot_boundaries_csv_path), exist_ok=True)
        shot_boundary_indices = generate_shot_boundary_indices(total_frame_count, cosine_similarities, threshold)
        shot_tuples = shot_boundary_indices_to_tuples(shot_boundary_indices)
        write_shot_boundary_tuples_to_csv(shot_tuples, shot_boundaries_csv_path)
        convert_frame_file_to_seconds(shot_boundaries_csv_path,video_path,shot_boundaries_csv_path)
