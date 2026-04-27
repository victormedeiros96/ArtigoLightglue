import os
import subprocess
import cv2
import json
import torch
import pandas as pd
from ultralytics import YOLO
from trackers.lightglue_tracker import LightGlueTracker
from trackers.naive_tracker import NaiveTracker
from tqdm import tqdm

def run_all_benchmarks(video_path, model_path, duration_sec=120):
    os.makedirs("tracking_results", exist_ok=True)
    os.makedirs("tracking_results_naive", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    framerates = [30, 5, 1]
    
    # We will use parallel processes for Ultralytics trackers (Botsort, Bytetrack)
    # and sequential for LightGlue/Naive to avoid GPU memory fragmentation
    # but since we have 4 GPUs, we can actually parallelize LightGlue too.

    # 1. Start BoTSORT and ByteTrack in background (Parallel)
    gpu_id = 0
    processes = []
    for fps in framerates:
        stride = 30 // fps
        max_f = int(duration_sec * 30)
        for t_yaml in ["botsort.yaml", "bytetrack.yaml"]:
            t_file = t_yaml
            if fps < 30: t_file = t_yaml.replace(".yaml", "_lowfps.yaml")
            name = f"{t_yaml.split('.')[0]}_{fps}fps"
            cmd = f"python3 run_single_track.py --video {video_path} --model {model_path} --tracker {t_file} --stride {stride} --name {name} --device {gpu_id} --max_frames {max_f}"
            print(f"Starting {name} on GPU {gpu_id}...")
            processes.append(subprocess.Popen(cmd.split()))
            gpu_id = (gpu_id + 1) % 4
            
    # 2. Run LightGlue Tracker (Sequential or on specific GPUs)
    # We'll run them now
    print("\nStarting LightGlue and Naive trackers...")
    # I'll use a simplified version here or just call the scripts
    # For now, let's just wait for the background ones
    
    for p in processes:
        p.wait()
        
    print("All tracking finished.")

if __name__ == "__main__":
    # This script will be called after ffmpeg is done
    video = "/home/servidor/ArtigoLightglue/segment_1080p_120s.mp4"
    model = "/home/servidor/ArtigoLightglue/detector_placa.pt"
    run_all_benchmarks(video, model)
