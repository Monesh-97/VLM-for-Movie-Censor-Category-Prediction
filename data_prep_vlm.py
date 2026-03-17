import pandas as pd
import cv2
import glob
import os
import json
from pathlib import Path

def extract_uniform_frames_to_video(mkv_path, out_mp4_path, num_frames=16):
    """
    Extracts `num_frames` uniformly across the entire length of the .mkv movie,
    and saves them as a short summary .mp4 video.
    """
    cap = cv2.VideoCapture(mkv_path)
    if not cap.isOpened():
        print(f"Error opening {mkv_path}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        print(f"Invalid frame count or FPS for {mkv_path}")
        cap.release()
        return False
        
    # Calculate step size to get exactly `num_frames`
    step = max(1, total_frames // num_frames)
    
    # Read the first frame to get dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
        
    height, width, _ = frame.shape
    
    # Initialize Mp4 Video Writer. We will write a video at 2 fps so the sequence is clear.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_mp4_path, fourcc, 2.0, (width, height))
    
    print(f"Extracting {num_frames} frames from {os.path.basename(mkv_path)}...")
    extracted = 0
    for i in range(num_frames):
        frame_idx = i * step
        # Ensure we don't go out of bounds
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            extracted += 1
            
    cap.release()
    out.release()
    print(f"Saved summary video to {out_mp4_path} ({extracted} frames)")
    return True

def prepare_data():
    dataset_dir = Path("Dataset")
    summary_dir = dataset_dir / "summary_videos"
    summary_dir.mkdir(exist_ok=True)
    
    df = pd.read_excel(dataset_dir / "dataset.xlsx")
    mkv_files = list(dataset_dir.glob("*.mkv"))
    
    dataset_json = []

    for idx, row in df.iterrows():
        movie_name = str(row['Movie Name']).lower().strip()
        cert = str(row['Certificate']).strip()
        
        # Find matching MKV file
        matching_file = None
        for mkv in mkv_files:
            # Check if movie_name is in the file name
            if movie_name in mkv.name.lower():
                matching_file = mkv
                break
                
        if matching_file is None:
            print(f"Warning: Could not find matching .mkv for movie '{movie_name}'")
            continue
            
        # Define output MP4 summary path
        out_mp4_name = f"{movie_name.replace(' ', '_')}_summary.mp4"
        out_mp4_path = summary_dir / out_mp4_name
        
        # Extract frames to create the summary if not already created
        if not out_mp4_path.exists():
            success = extract_uniform_frames_to_video(str(matching_file), str(out_mp4_path), num_frames=16)
            if not success:
                continue
                
        # Create LLaVA conversational format JSON entry
        # The prompt asks the model to watch the video and output the certificate
        entry = {
            "id": f"movie_{idx}",
            "video": str(out_mp4_path),
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\nAnalyze the context, themes, and content of this movie summary. Based on your understanding, what is the appropriate CBFC (India) movie certificate? The expected classes are ['U', 'U/A', 'A']. Output only the certificate."
                },
                {
                    "from": "gpt",
                    "value": f"{cert}"
                }
            ]
        }
        dataset_json.append(entry)
        
    with open("vl_train_data.json", "w") as f:
        json.dump(dataset_json, f, indent=4)
        
    print(f"Data preparation complete! Created 'vl_train_data.json' with {len(dataset_json)} samples.")

if __name__ == "__main__":
    prepare_data()
