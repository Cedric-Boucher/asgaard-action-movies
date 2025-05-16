from transformers.models.siglip import SiglipVisionModel
from transformers.models.auto.processing_auto import AutoProcessor
import torch
import argparse
import cv2
from PIL import Image
import os
import csv
from tqdm import tqdm
from ffmpeg import FFmpeg
import json
from collections.abc import Iterable, Generator
from typing import Optional


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COSINE_SIMILARITY_CSV_DIRECTORY: str = "./cosine_similarities/"

os.makedirs(COSINE_SIMILARITY_CSV_DIRECTORY, exist_ok=True)

def video_to_pil_frames(video_path) -> Generator[Image.Image, None, None]:
    video = cv2.VideoCapture(video_path)
    total_frames: int = video_total_frame_count(video_path)

    for _ in tqdm(range(total_frames), desc="Fetching Frames", total=total_frames):
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image: Image.Image = Image.fromarray(frame_rgb)

        yield pil_image  # Yield one frame at a time

    video.release()

def generate_frame_vectors(video_path: str, model_id="google/siglip-base-patch16-224") -> Generator[torch.Tensor, None, None]:
    print(f"Using checkpoint:{model_id}")
    model: SiglipVisionModel = SiglipVisionModel.from_pretrained(model_id,device_map=DEVICE)
    processor = AutoProcessor.from_pretrained(model_id)
    frames: Generator[Image.Image, None, None] = video_to_pil_frames(video_path)
    total_frames: int = video_total_frame_count(video_path)
    for frame in tqdm(frames,desc="Generating Vectors", total=total_frames):
        inputs = processor(images=frame, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        for tensor in outputs.pooler_output:
            yield tensor.detach().flatten().reshape(1,-1)

def compute_cosine_similarity(frames: Iterable[torch.Tensor], total_frame_count: int) -> Generator[tuple[int, int, float], None, None]:
    last_frame: Optional[torch.Tensor] = None
    for i, frame in enumerate(tqdm(frames, desc="Computing Cosine Similarities", total=total_frame_count)):
        if last_frame is None:
            last_frame = frame
            continue

        assert frame is not None
        assert last_frame is not None
        similarity: int | float | bool = torch.nn.functional.cosine_similarity(last_frame, frame).item()
        assert isinstance(similarity, float)
        yield (i-1, i, float(similarity))

        last_frame = frame

def cut_video(video_path, frame_ranges, write_video=False):
    scene_list = []
    scene_list_seconds = []
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), base_name)
    os.makedirs(output_dir, exist_ok=True)
    probe = json.loads(FFmpeg(executable="ffprobe").input(video_path, print_format="json", show_streams=None).execute())
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    frame_rate = eval(video_streams[0]['r_frame_rate'])

    for i, (start_frame, end_frame) in enumerate(frame_ranges):
        start_time = start_frame / frame_rate
        start_time_seconds = convert_seconds(start_time)
        end_time = (end_frame + 1) / frame_rate
        end_time_seconds = convert_seconds(end_time)
        scene_list.append((start_time,end_time))
        scene_list_seconds.append((start_time_seconds,end_time_seconds))
        output_file = os.path.join(output_dir, f"{base_name}_part_{i+1}.mp4")
        if write_video:
            FFmpeg().option("y").input(video_path, ss=start_time, to=end_time).output(output_file).execute()
    return scene_list,scene_list_seconds

def overlay_markers(video_path, shot_boundaries, output_path):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_path, exist_ok=True)
    total_frames: int = video_total_frame_count(video_path)
    for frame_idx in tqdm(range(total_frames),desc="Writing Frames to Disk:", total=total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        for start, end in shot_boundaries:
            if frame_idx == start or frame_idx == end:
                height, width, _ = frame.shape
                cv2.line(frame, (0, height//2), (width, height//2), (0, 0, 255), 5)
                cv2.putText(frame, "Shot Boundary", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imwrite(f"{output_path}/frame_{frame_idx}.jpg", frame)
    cap.release()

def get_key_shot_frame(shot_boundaries,frame_vectors):
    for start,end in shot_boundaries:
        frame_vectors_interest = frame_vectors[start:end+1]
        frame_vectors_mean = torch.sum(frame_vectors_interest,dim=0) / len(frame_vectors_interest)
        frame_vector_distances = torch.norm(frame_vectors_mean - frame_vectors_interest, dim=1)
        _, min_idx = torch.min(frame_vector_distances, 0)
        yield (start + min_idx).item()

def write_key_shots(key_shots,output_path):
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Shot"])  # Header row
        for shot in key_shots:  # Iterate over the generator
            writer.writerow([shot])

def fetch_key_shots(key_shot_frames,video_path):
    print(f"Fetching Frames: {video_path}")
    video = cv2.VideoCapture(video_path)
    total_frames: int = video_total_frame_count(video_path)
    print(f"Total Frames: {total_frames}")

    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i in key_shot_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            yield pil_image
        else:
            continue
    video.release()

def convert_seconds(seconds: float | int) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds: float | int = seconds % 60  # Keep decimal precision if needed

    return f"{hours:02}:{minutes:02}:{remaining_seconds:06.3f}"

def video_total_frame_count(video_path: str) -> int:
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return total_frames

def write_cosine_similarities_to_csv(cosine_similarities: Iterable[tuple[int, int, float]], output_path: str) -> None:
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["First Frame Index", "Second Frame Index", "Cosine Similarity"])
        for first_frame_index, second_frame_index, cosine_similarity in cosine_similarities:
            assert isinstance(first_frame_index, int)
            assert isinstance(second_frame_index, int)
            assert isinstance(cosine_similarity, float)
            writer.writerow((first_frame_index, second_frame_index, cosine_similarity))
            file.flush()


if __name__=="__main__":
    print(f"device:{DEVICE}")
    parser = argparse.ArgumentParser(description="Shot Generator Using Cosine Similarity")
    parser.add_argument('--video', required=True, type=str, help="Video path")
    parser.add_argument("--write-frames", default=None, type=str, help="Write Frames to Files")
    parser.add_argument("--key-shots", action="store_true",help="Get Key Shots")
    args = parser.parse_args()
    video_path = args.video
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frame_count: int = video_total_frame_count(video_path)
    frames = generate_frame_vectors(video_path)
    similarities = compute_cosine_similarity(frames, total_frame_count)
    cosine_similarities_csv_path: str = os.path.join(COSINE_SIMILARITY_CSV_DIRECTORY, base_name+"_cosine_similarities.csv")
    write_cosine_similarities_to_csv(similarities, cosine_similarities_csv_path)
    if False:
        if args.write_frames is not None:
            os.makedirs(os.path.dirname(args.write_frames), exist_ok=True)
            overlay_markers(video_path, shot_tuples, base_name)
        if args.key_shots:
            key_shots = get_key_shot_frame(shot_tuples,frame_vectors=torch.cat(list(generate_frame_vectors(video_path))))
            write_key_shots(key_shots,base_name+"_shots.csv")
