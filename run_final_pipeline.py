import os
import cv2
import json
import torch
import pandas as pd
from ultralytics import YOLO
from trackers.lightglue_tracker import LightGlueTracker
from tqdm import tqdm

def run_benchmark(video_path, model_path, fps_list, duration_sec=300):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    base_fps = 30
    
    final_summary = []
    
    for fps in fps_list:
        stride = base_fps // fps
        print(f"\n>>> Running Final Methodology: LightGlue Tracker at {fps}fps")
        
        # Reset Tracker for each FPS run
        # ---------------------------------------------------------
        # EQUAÇÃO ADAPTATIVA DE PARÂMETROS (PROPOSTA DO ARTIGO)
        # ---------------------------------------------------------
        # max_age: 1fps -> 1 frame | 30fps -> 15 frames (0.5s)
        dynamic_max_age = int(1 + (14 * (fps - 1) / 29))
        # min_matches: 1fps -> 8 pts | 30fps -> 3 pts
        dynamic_min_matches = int(8 - (5 * (fps - 1) / 29))
        
        tracker = LightGlueTracker(
            device=device, 
            max_age=dynamic_max_age, 
            min_matches_short=dynamic_min_matches,
            min_matches_global=dynamic_min_matches + 4, # ReID sempre um pouco mais rigoroso
            verbose=False
        )
        print(f"Propriedades Adaptativas para {fps}fps: Age={dynamic_max_age}, MinMatches={dynamic_min_matches}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
            
        max_proc_frames = int(duration_sec * base_fps)
        
        results_data = []
        pbar = tqdm(total=max_proc_frames // stride, desc=f"LG_{fps}fps")
        
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or f_idx >= max_proc_frames:
                break
            
            if f_idx % stride == 0:
                # 1. Detection
                res = model.predict(frame, imgsz=1280, verbose=False, classes=list(range(11)))[0]
                
                bboxes, crops, classes = [], [], []
                if res.boxes is not None:
                    det_boxes = res.boxes.xyxy.cpu().numpy()
                    det_classes = res.boxes.cls.cpu().numpy().astype(int)
                    for box, cls in zip(det_boxes, det_classes):
                        x1, y1, x2, y2 = map(int, box)
                        pad = 15
                        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                        cx2, cy2 = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            bboxes.append(box)
                            crops.append(crop)
                            classes.append(cls)
                
                # 2. Update LightGlue Tracker with Class and Frame information
                active_tracks = tracker.update(bboxes, crops, classes, f_idx)
                
                for trk in active_tracks:
                    results_data.append({
                        "frame": f_idx,
                        "id": trk['id'],
                        "box": trk['bbox'].tolist()
                    })
                pbar.update(1)
            f_idx += 1
            
        cap.release()
        pbar.close()
        
        # Save JSON
        os.makedirs("tracking_results", exist_ok=True)
        json_out = f"tracking_results/lightglue_{fps}fps.json"
        with open(json_out, "w") as f:
            json.dump(results_data, f)
            
        unique_ids = len(set(d['id'] for d in results_data))
        total_detections = len(results_data)
        
        final_summary.append({
            "Methodology": "Proposed (LightGlue)",
            "FPS": f"{fps}fps",
            "Unique IDs": unique_ids,
            "Sightings": total_detections
        })

    return pd.DataFrame(final_summary)

if __name__ == "__main__":
    # Using the 1080p segment for speed, but extended to 5 minutes
    # If segment_1080p.mp4 is only 1 min, we should recreate it or use original
    video = "/home/servidor/ArtigoLightglue/segment_1080p_120s.mp4"
    model_path = "/home/servidor/ArtigoLightglue/detector_placa.pt"
    
    # Let's check if the video has enough duration, otherwise use original (which is 5K but we can resize in predict)
    # Actually, for the paper, let's use the original video but imgsz=1280 in YOLO.
    # LightGlue will crop the ROI so it's efficient.
    
    df = run_benchmark(video, model_path, [30, 5, 1], duration_sec=120) # 2 minutes for now to be safe
    
    print("\n--- EXTENDED RESEARCH RESULTS (2 MIN) ---")
    print(df.to_markdown(index=False))
    df.to_csv("/home/servidor/ArtigoLightglue/results/extended_results.csv", index=False)
