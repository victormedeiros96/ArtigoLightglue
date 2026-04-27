import torch
from ultralytics import YOLO
from trackers.lightglue_tracker import LightGlueTracker
import cv2
import json

def test():
    print("Iniciando teste de 1fps...")
    model = YOLO("/home/servidor/ArtigoLightglue/detector_placa.pt")
    tracker = LightGlueTracker(device='cuda', max_age=1, min_matches_short=8)
    video = "/home/servidor/ArtigoLightglue/proxies/GX010084.MP4"
    cap = cv2.VideoCapture(video)
    
    data = []
    for f_idx in range(3600):
        ret, frame = cap.read()
        if not ret: break
        
        if f_idx % 30 == 0:
            res = model.predict(frame, imgsz=1280, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                bboxes = res.boxes.xyxy.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                crops = []
                for b in bboxes:
                    x1, y1, x2, y2 = map(int, b)
                    crops.append(frame[y1:y2, x1:x2])
                
                # O Erro deve estar aqui:
                tracks = tracker.update(bboxes, crops, classes, f_idx)
                for t in tracks:
                    data.append({"f": f_idx, "id": t["id"], "cls": int(t["cls"])})
            if (f_idx // 30) % 10 == 0:
                print(f"Processado segundo: {f_idx // 30}")
                
    cap.release()
    print(f"Sucesso! Total de avistamentos: {len(data)}")

if __name__ == "__main__":
    test()
