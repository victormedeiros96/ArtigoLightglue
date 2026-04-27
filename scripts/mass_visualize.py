import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from trackers.lightglue_tracker import LightGlueTracker
from collections import defaultdict, deque
from tqdm import tqdm

# Para os trackers padrões da Ultralytics
class UltralyticsVisualizer:
    def __init__(self, model_path, tracker_type, stride, device):
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        self.stride = stride
        self.device = device
        
    def get_tracks(self, frame):
        # Usando o track nativo para BoTSORT e ByteTrack
        results = self.model.track(frame, tracker=self.tracker_type, persist=True, verbose=False, device=self.device, imgsz=1280)[0]
        tracks = []
        if results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy()
            bboxes = results.boxes.xyxy.cpu().numpy()
            for tid, box in zip(ids, bboxes):
                tracks.append({"id": int(tid), "bbox": box})
        return tracks

def draw_premium_frame(frame, tracks, trails, colors, info_text):
    for trk in tracks:
        tid = trk['id']
        box = trk['bbox']
        x1, y1, x2, y2 = map(int, box)
        color = colors[tid % 1000]
        
        # Center for trail
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        trails[tid].append(center)
        
        # Draw Trail (fading effect)
        if len(trails[tid]) > 1:
            for i in range(1, len(trails[tid])):
                alpha = i / len(trails[tid])
                cv2.line(frame, trails[tid][i-1], trails[tid][i], color, 2)
        
        # Draw Neon Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Glow effect
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), color, 1)
        
        # Label with background
        label = f"ID: {tid}"
        (l_w, l_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - l_h - 10), (x1 + l_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Info Overlay
    cv2.rectangle(frame, (10, 10), (450, 70), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

def run_mass_visualization():
    video_path = "/home/servidor/ArtigoLightglue/segment_1080p.mp4"
    model_path = "/home/servidor/ArtigoLightglue/detector_placa.pt"
    output_root = "/home/servidor/ArtigoLightglue/results/videos"
    os.makedirs(output_root, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    framerates = [30, 5, 1]
    # Algoritmos a testar
    algorithms = ["botsort.yaml", "bytetrack.yaml", "lightglue", "naive"]
    
    colors = np.random.randint(0, 255, (1000, 3)).tolist()
    
    for fps in framerates:
        stride = 30 // fps
        for alg in algorithms:
            out_name = f"{alg.split('.')[0]}_{fps}fps.mp4"
            out_path = os.path.join(output_root, out_name)
            
            print(f"\nRendering: {out_name}")
            cap = cv2.VideoCapture(video_path)
            w, h = int(cap.get(3)), int(cap.get(4))
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h)) # Keep 30fps for smooth viewing
            
            trails = defaultdict(lambda: deque(maxlen=20))
            
            # Init Trackers
            if alg == "lightglue":
                tracker = LightGlueTracker(device=device)
                model = YOLO(model_path)
            elif alg == "naive":
                from trackers.naive_tracker import NaiveTracker # Assuming we named it so
                tracker = NaiveTracker(iou_threshold=0.3)
                model = YOLO(model_path)
            else:
                extra_v = UltralyticsVisualizer(model_path, alg, stride, device)
            
            for f_idx in tqdm(range(600), desc=out_name): # 20 seconds segment
                ret, frame = cap.read()
                if not ret: break
                
                # Only process on stride
                if f_idx % stride == 0:
                    if alg in ["lightglue", "naive"]:
                        res = model.predict(frame, imgsz=1280, verbose=False)[0]
                        bboxes, crops = [], []
                        if res.boxes is not None:
                            for b in res.boxes.xyxy.cpu().numpy():
                                bboxes.append(b); x1,y1,x2,y2 = map(int, b)
                                crops.append(frame[y1:y2, x1:x2])
                        
                        if alg == "lightglue":
                            tracks = tracker.update(bboxes, crops)
                        else:
                            # Naive tracking
                            ids = tracker.update(bboxes)
                            tracks = [{"id": i, "bbox": b} for i, b in zip(ids, bboxes)]
                    else:
                        tracks = extra_v.get_tracks(frame)
                else:
                    # On other frames, just show previous tracks or clear
                    # For visual comparison, we only show dots/text when detected
                    tracks = [] 
                
                info = f"{alg.upper()} | {fps} FPS"
                frame = draw_premium_frame(frame, tracks, trails, colors, info)
                out.write(frame)
                
            cap.release()
            out.release()

if __name__ == "__main__":
    run_mass_visualization()
