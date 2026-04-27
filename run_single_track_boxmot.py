"""
Runner de trackers boxmot (StrongSORT, OC-SORT, DeepOCSORT, HybridSORT, SFSort, BoostTrack)
Usa o detector YOLO para detecção e boxmot para tracking.
"""
import argparse
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import os
from ultralytics import YOLO

def get_tracker(tracker_name, device):
    """Instancia o tracker correto usando a API do boxmot v17"""
    try:
        from boxmot.trackers.tracker_zoo import create_tracker
    except ImportError:
        raise ImportError("boxmot nao encontrado! Instale com: pip install boxmot")

    reid_weights = "osnet_x0_25_msmt17.pt" if tracker_name in ["strongsort", "deepocsort", "hybridsort", "boosttrack"] else None
    
    return create_tracker(
        tracker_type=tracker_name,
        tracker_config=None,
        reid_weights=reid_weights,
        device=device,
        half=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--model")
    parser.add_argument("--tracker")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--name")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max_frames", type=int, default=3600)
    args = parser.parse_args()

    results_dir = "/home/servidor/ArtigoLightglue/tracking_results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{args.name}.json")

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else "cpu")

    # Carrega detector YOLO
    yolo = YOLO(args.model)

    # Carrega tracker boxmot
    tracker = get_tracker(args.tracker, device)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    data = []
    f_idx = 0
    processed = 0
    max_to_process = args.max_frames // args.stride

    pbar = tqdm(total=max_to_process, desc=args.name)

    while cap.isOpened() and f_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret: break

        if f_idx % args.stride == 0:
            # Detectar com YOLO
            res = yolo.predict(frame, imgsz=1280, verbose=False,
                               device=args.device, classes=list(range(11)))[0]

            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()     # [N, 4] xyxy
                confs = res.boxes.conf.cpu().numpy()     # [N]
                clss  = res.boxes.cls.cpu().numpy()      # [N]

                # boxmot espera [x1, y1, x2, y2, conf, cls]
                dets = np.column_stack([boxes, confs, clss])
            else:
                dets = np.empty((0, 6))

            # Atualizar tracker
            tracks = tracker.update(dets, frame)  # retorna [x1, y1, x2, y2, id, conf, cls, ...]

            if tracks is not None and len(tracks) > 0:
                for t in tracks:
                    tid = int(t[4])
                    box = t[:4].tolist()
                    data.append({"frame": f_idx, "id": tid, "box": box})

            processed += 1
            pbar.update(1)

        f_idx += 1

    cap.release()
    pbar.close()

    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Finished {args.name}")


if __name__ == "__main__":
    main()
