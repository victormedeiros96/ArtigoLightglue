import argparse
from ultralytics import YOLO
import json
from tqdm import tqdm
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--model")
    parser.add_argument("--tracker")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--name")
    parser.add_argument("--device", type=int)
    parser.add_argument("--max_frames", type=int, default=3600) # 2 mins @ 30fps
    args = parser.parse_args()

    results_dir = "/home/servidor/ArtigoLightglue/mass_results/jsons"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{args.name}.json")

    model = YOLO(args.model)
    # We want to process up to max_frames IN THE ORIGINAL VIDEO
    # So the generator should stop after i * stride >= max_frames
    results = model.track(source=args.video, 
                          tracker=args.tracker, 
                          vid_stride=args.stride, 
                          device=args.device, 
                          persist=True, 
                          stream=True, 
                          verbose=False,
                          imgsz=1280,
                          classes=list(range(11)))

    data = []
    # Total frames to process in the loop
    total_to_process = args.max_frames // args.stride
    
    for i, res in enumerate(tqdm(results, desc=args.name, total=total_to_process)):
        frame_orig = i * args.stride
        if frame_orig >= args.max_frames:
            break
            
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            for obj_id, box in zip(ids, boxes):
                data.append({"frame": frame_orig, "id": int(obj_id), "box": box.tolist()})

    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Finished {args.name}")

if __name__ == "__main__":
    main()
