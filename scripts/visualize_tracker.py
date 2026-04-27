import cv2
import torch
import numpy as np
from ultralytics import YOLO
from trackers.lightglue_tracker import LightGlueTracker
from collections import defaultdict, deque
from tqdm import tqdm

def main():
    video_path = "/home/servidor/ArtigoLightglue/segment_1080p.mp4"
    model_path = "/home/servidor/ArtigoLightglue/detector_placa.pt"
    output_path = "/home/servidor/ArtigoLightglue/results/visual_validation.mp4"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    tracker = LightGlueTracker(device=device)
    
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # To store trails: id -> deque of (x, y)
    trails = defaultdict(lambda: deque(maxlen=30))
    
    # Colors for different IDs
    colors = np.random.randint(0, 255, (1000, 3)).tolist()

    print(f"Generating visual validation video at {output_path}...")
    
    # We'll process first 600 frames (20s) for a quick demo
    max_frames = 600
    
    for f_idx in tqdm(range(max_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        # Detection
        res = model.predict(frame, imgsz=1280, verbose=False)[0]
        
        bboxes = []
        crops = []
        if res.boxes is not None:
            det_boxes = res.boxes.xyxy.cpu().numpy()
            for box in det_boxes:
                x1, y1, x2, y2 = map(int, box)
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    bboxes.append(box)
                    crops.append(crop)
        
        # Update Tracker
        active_tracks = tracker.update(bboxes, crops)
        
        # Draw Results
        for trk in active_tracks:
            tid = trk['id']
            box = trk['bbox']
            x1, y1, x2, y2 = map(int, box)
            
            # Center of the box for the trail
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            trails[tid].append(center)
            
            color = colors[tid % 1000]
            
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # Draw ID
            cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw Trail
            if len(trails[tid]) > 1:
                for i in range(1, len(trails[tid])):
                    cv2.line(frame, trails[tid][i-1], trails[tid][i], color, 2)
        
        # Overlay Current Frame Info
        cv2.putText(frame, f"Frame: {f_idx} | Method: LightGlue Tracker", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    cap.release()
    out.release()
    print("\nVisual validation video finished!")

if __name__ == "__main__":
    main()
