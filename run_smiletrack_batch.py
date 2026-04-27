#!/usr/bin/env python3
"""Batch runner for SMILEtrack using an Ultralytics YOLO detector."""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent
SMILETRACK_YOLOV7_DIR = REPO_ROOT / "external" / "SMILEtrack" / "SMILEtrack_Official" / "yolov7"

if not SMILETRACK_YOLOV7_DIR.exists():
    raise FileNotFoundError(
        f"SMILEtrack not found at {SMILETRACK_YOLOV7_DIR}. "
        "Clone repo first: https://github.com/WWangYuHsiang/SMILEtrack"
    )

if str(SMILETRACK_YOLOV7_DIR) not in sys.path:
    sys.path.insert(0, str(SMILETRACK_YOLOV7_DIR))

from tracker.mc_SMILEtrack import SMILEtrack  # noqa: E402


DEFAULT_VIDEOS = [
    REPO_ROOT / "proxies" / "GX010084.MP4",
    REPO_ROOT / "proxies" / "GX010083.MP4",
    REPO_ROOT / "proxies" / "GX010069.MP4",
    REPO_ROOT / "proxies" / "GX010076.MP4",
    REPO_ROOT / "proxies" / "GX010086.MP4",
    REPO_ROOT / "proxies" / "GX010067.MP4",
    REPO_ROOT / "proxies" / "GX010080.MP4",
    REPO_ROOT / "proxies" / "GX010081.MP4",
    REPO_ROOT / "proxies" / "GX010085.MP4",
]


def build_tracker_args(args: argparse.Namespace, video_name: str, frame_rate: float) -> SimpleNamespace:
    return SimpleNamespace(
        # Meta args used by tracker internals
        name=video_name,
        ablation=False,
        mot20=False,
        with_reid=False,
        fast_reid_config="",
        fast_reid_weights="",
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=args.appearance_thresh,
        cmc_method=args.cmc_method,
        # Tracking args
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        min_box_area=args.min_box_area,
        frame_rate=frame_rate,
    )


def color_for_track(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_track(frame: np.ndarray, tlbr: np.ndarray, track_id: int, cls_id: int, score: float) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in np.round(tlbr)]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    color = color_for_track(track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"id:{track_id} cls:{cls_id} {score:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    text_y = max(y1 - 8, text_h + 2)
    cv2.rectangle(frame, (x1, text_y - text_h - 4), (x1 + text_w + 6, text_y + 2), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def build_detections(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    dets = np.column_stack([boxes_xyxy, confs, classes])
    return dets.astype(np.float32, copy=False)


def run_video(
    video_path: Path,
    output_path: Path,
    model: YOLO,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")

    tracker_args = build_tracker_args(args, video_path.stem, fps)
    tracker = SMILEtrack(tracker_args, frame_rate=fps)

    if args.max_frames is not None and args.max_frames > 0 and total_frames > 0:
        progress_total = min(total_frames, args.max_frames)
    else:
        progress_total = total_frames if total_frames > 0 else None
    progress = tqdm(total=progress_total, desc=video_path.name, unit="frame")

    processed = 0
    track_draws = 0
    while True:
        if args.max_frames is not None and args.max_frames > 0 and processed >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            classes=args.classes,
            verbose=False,
        )[0]
        detections = build_detections(result)

        tracks = tracker.update(detections, frame)
        frame_track_draws = 0
        for track in tracks:
            tlwh = track.tlwh
            if float(tlwh[2] * tlwh[3]) < args.min_box_area:
                continue
            draw_track(frame, track.tlbr, int(track.track_id), int(track.cls), float(track.score))
            frame_track_draws += 1
            track_draws += 1

        writer.write(frame)
        processed += 1
        progress.update(1)
        progress.set_postfix(dets=int(len(detections)), tracks=frame_track_draws, total_tracks=track_draws)

        if args.stop_after_detections is not None and args.stop_after_detections > 0:
            if track_draws >= args.stop_after_detections:
                break

    progress.close()
    cap.release()
    writer.release()
    return processed, total_frames, track_draws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SMILEtrack over videos using an Ultralytics detector.")
    parser.add_argument("--model", default=str(REPO_ROOT / "detector_placa.pt"), help="Path to detector .pt model.")
    parser.add_argument(
        "--videos",
        nargs="*",
        default=[str(v) for v in DEFAULT_VIDEOS],
        help="Input videos. If omitted, runs the predefined 9 proxy videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "videos_smiletrack"),
        help="Directory where output videos are saved.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="Detector NMS IoU threshold.")
    parser.add_argument("--device", default="", help="Inference device for Ultralytics (e.g. 0, 0,1, cpu).")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Optional class filter.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most this many frames from each input video.",
    )
    parser.add_argument(
        "--stop-after-detections",
        type=int,
        default=None,
        help="Stop video early once this total number of tracked boxes has been drawn.",
    )

    # SMILEtrack args
    parser.add_argument("--track_high_thresh", type=float, default=0.3)
    parser.add_argument("--track_low_thresh", type=float, default=0.05)
    parser.add_argument("--new_track_thresh", type=float, default=0.4)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.7)
    parser.add_argument("--min_box_area", type=float, default=10.0)
    parser.add_argument("--cmc-method", default="sparseOptFlow")
    parser.add_argument("--proximity_thresh", type=float, default=0.5)
    parser.add_argument("--appearance_thresh", type=float, default=0.25)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [Path(v).expanduser().resolve() for v in args.videos]
    missing = [str(v) for v in videos if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing input video(s):\n" + "\n".join(missing))

    print(f"Loading detector: {model_path}")
    model = YOLO(str(model_path))

    print(f"Saving outputs to: {output_dir}")
    for video in videos:
        out_path = output_dir / f"{video.stem}_smiletrack.mp4"
        print(f"\n[RUN] {video.name} -> {out_path.name}")
        processed, total, drawn = run_video(video, out_path, model, args)
        suffix = f"/{total}" if total > 0 else ""
        print(f"[DONE] {video.name}: {processed}{suffix} frames, {drawn} tracks drawn")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
