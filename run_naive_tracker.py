import os
import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from trackers.naive_tracker import NaiveTracker

def run_naive_benchmark(video_path, model_path, fps_list, duration_sec=120):
    model = YOLO(model_path)
    results_dir = "/home/servidor/ArtigoLightglue/tracking_results_naive"
    os.makedirs(results_dir, exist_ok=True)
    base_fps = 30
    max_frames = int(duration_sec * base_fps)
    
    for fps in fps_list:
        stride = base_fps // fps
        tracker = NaiveTracker(iou_threshold=0.3)
        cap = cv2.VideoCapture(video_path)
        
        data = []
        f_idx = 0
        pbar = tqdm(total=max_frames // stride, desc=f"Naive_{fps}fps")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or f_idx >= max_frames: break
            
            if f_idx % stride == 0:
                res = model.predict(frame, imgsz=1280, verbose=False, classes=list(range(11)))[0]
                if res.boxes is not None:
                    boxes = res.boxes.xyxy.cpu().numpy().tolist()
                    ids = tracker.update(boxes)
                    for obj_id, box in zip(ids, boxes):
                        data.append({"frame": f_idx, "id": obj_id, "box": box})
                pbar.update(1)
            f_idx += 1
        cap.release()
        pbar.close()
        
        with open(os.path.join(results_dir, f"naive_{fps}fps.json"), "w") as f:
            json.dump(data, f)

if __name__ == "__main__":
    import cv2
    video = "/home/servidor/ArtigoLightglue/segment_1080p_120s.mp4"
    model = "/home/servidor/ArtigoLightglue/detector_placa.pt"
    run_naive_benchmark(video, model, [30, 5, 1])
