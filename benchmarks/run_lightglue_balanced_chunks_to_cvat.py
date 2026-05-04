import argparse
import json
import math
import os
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.lightglue_tracker import LightGlueTracker


ROOT = Path("/home/servidor/ArtigoLightglue")
MODEL_PATH = ROOT / "models/detector_placa.pt"
BALANCED_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_plate_balanced_frames")
BALANCED_RESULTS_ROOT = ROOT / "results/plate_balanced_chunks"
OUTPUT_ROOT = ROOT / "results/cvat_lightglue_balanced_chunks"
XML_DIR = OUTPUT_ROOT / "xmls"
JSON_DIR = OUTPUT_ROOT / "jsons"
TEMPLATE_DIR = OUTPUT_ROOT / "templates"

CVAT_HOST = "http://192.168.18.140"
CVAT_PORT = "8080"
CVAT_ORG = "RDT3"
CVAT_PROJECT_ID = 79
CVAT_USER = "superrdt"
CVAT_PASS = "superpwdrdt"


def ensure_dirs():
    for path in [OUTPUT_ROOT, XML_DIR, JSON_DIR, TEMPLATE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=merged_env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def cvat_base_cmd(args):
    return [
        "cvat-cli",
        "--auth",
        args.cvat_user,
        "--server-host",
        args.cvat_host,
        "--server-port",
        args.cvat_port,
        "--organization",
        args.cvat_org,
    ]


def get_task_map(args):
    output = run_cmd(cvat_base_cmd(args) + ["task", "ls", "--json"], env={"PASS": args.cvat_pass})
    tasks = json.loads(output)
    mapping = {}
    for task in tasks:
        if task.get("project_id") != args.project_id:
            continue
        name = task.get("name", "")
        if name.startswith(args.task_prefix):
            mapping[name] = int(task["id"])
    return mapping


def load_manifest(chunk_name, args):
    path = Path(args.results_root) / f"{chunk_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest nao encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for row in data["manifest"]:
        selected_path = Path(row["selected_path"])
        if not selected_path.exists():
            selected_path = Path(args.balanced_root) / chunk_name / row["file_name"]
        rows.append(
            {
                "task_frame": int(row["task_frame"]),
                "video": row["video"],
                "original_frame": int(row["original_frame"]),
                "path": selected_path,
                "file_name": row["file_name"],
            }
        )
    return sorted(rows, key=lambda item: item["task_frame"])


def new_tracker(device, next_id):
    tracker = LightGlueTracker(device=device, accept_th=1.2, motion_weight=0.3)
    tracker.next_id = next_id
    return tracker


def predict_kwargs(args):
    kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
        "classes": list(range(11)),
    }
    if args.device != "auto":
        kwargs["device"] = args.device
    return kwargs


def run_tracking(chunk_name, model, args):
    items = load_manifest(chunk_name, args)
    tracker = new_tracker(args.device, 1)
    next_id = 1
    prev_video = None
    prev_original_frame = None
    tracking_data = []

    pbar = tqdm(items, desc=chunk_name)
    for item in pbar:
        video = item["video"]
        original_frame = int(item["original_frame"])
        task_frame = int(item["task_frame"])

        if (
            prev_video is not None
            and (video != prev_video or original_frame - prev_original_frame > args.reset_gap)
        ):
            next_id = tracker.next_id
            tracker = new_tracker(args.device, next_id)

        frame = cv2.imread(str(item["path"]))
        if frame is None:
            print(f"[WARN] Nao consegui ler: {item['path']}")
            prev_video = video
            prev_original_frame = original_frame
            continue

        det_res = model.predict(frame, **predict_kwargs(args))[0]
        det_bboxes = []
        det_classes = []
        if det_res.boxes is not None:
            det_bboxes = det_res.boxes.xyxy.cpu().numpy()
            det_classes = det_res.boxes.cls.cpu().numpy().astype(int)

        active_tracks = tracker.update(det_bboxes, frame, det_classes, original_frame, stride=1)
        for trk in active_tracks:
            tracking_data.append(
                {
                    "frame": task_frame,
                    "video": video,
                    "original_frame": original_frame,
                    "id": int(trk["id"]),
                    "box": [float(x) for x in trk["bbox"]],
                    "cls": int(trk["cls"]),
                    "file_name": item["file_name"],
                }
            )

        prev_video = video
        prev_original_frame = original_frame

    pbar.close()
    return tracking_data, len(items)


def filter_short_tracks(tracking_data, fps_target=30):
    id_counts = defaultdict(int)
    for row in tracking_data:
        id_counts[row["id"]] += 1
    min_presence = max(1, math.ceil(0.3 * fps_target))
    return [row for row in tracking_data if id_counts[row["id"]] >= min_presence]


def export_task_template(task_id, chunk_name, args):
    out_zip = TEMPLATE_DIR / f"{chunk_name}_task{task_id}_template.zip"
    if out_zip.exists():
        out_zip.unlink()
    run_cmd(
        cvat_base_cmd(args)
        + ["task", "export-dataset", str(task_id), str(out_zip), "--format", "CVAT for images 1.1"],
        env={"PASS": args.cvat_pass},
    )
    with zipfile.ZipFile(out_zip) as zf:
        return ET.fromstring(zf.read("annotations.xml"))


def clear_existing_annotations(root):
    for image_elem in root.findall("image"):
        for child in list(image_elem):
            image_elem.remove(child)
    for track_elem in list(root.findall("track")):
        root.remove(track_elem)


def extract_label_names(root):
    nodes = root.findall("./meta/task/labels/label/name")
    if not nodes:
        nodes = root.findall("./meta/job/labels/label/name")
    return [(node.text or "").strip() for node in nodes]


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(0.0, min(float(width - 1), float(x2)))
    y2 = max(0.0, min(float(height - 1), float(y2)))
    if x2 <= x1:
        x2 = min(float(width - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height - 1), y1 + 1.0)
    return [x1, y1, x2, y2]


def group_tracks(clean_data):
    grouped = defaultdict(list)
    for row in clean_data:
        grouped[row["id"]].append(row)
    for track_id in grouped:
        grouped[track_id].sort(key=lambda item: item["frame"])
    return grouped


def majority_class(track_obs):
    counts = defaultdict(int)
    for obs in track_obs:
        counts[obs["cls"]] += 1
    return max(counts.items(), key=lambda item: item[1])[0]


def add_outside_box(track_elem, frame, box):
    ET.SubElement(
        track_elem,
        "box",
        {
            "frame": str(frame),
            "keyframe": "1",
            "outside": "1",
            "occluded": "0",
            "xtl": f"{box[0]:.2f}",
            "ytl": f"{box[1]:.2f}",
            "xbr": f"{box[2]:.2f}",
            "ybr": f"{box[3]:.2f}",
            "z_order": "0",
        },
    )


def build_track_elements(root, clean_data):
    images = root.findall("image")
    if not images:
        raise RuntimeError("Template XML sem nós <image>.")
    width = int(images[0].attrib.get("width", "1920"))
    height = int(images[0].attrib.get("height", "1080"))
    stop_frame = max(int(img.attrib["id"]) for img in images)
    labels = extract_label_names(root)
    if not labels:
        raise RuntimeError("Nao foi possivel ler labels do template XML.")

    created = 0
    skipped = 0
    for track_id, obs in sorted(group_tracks(clean_data).items()):
        cls_id = majority_class(obs)
        if cls_id < 0 or cls_id >= len(labels):
            skipped += 1
            continue
        track_elem = ET.SubElement(
            root,
            "track",
            {"id": str(track_id), "label": labels[cls_id], "source": "auto"},
        )
        prev_frame = None
        prev_box = None
        for item in obs:
            frame = int(item["frame"])
            box = clamp_box(item["box"], width, height)
            if prev_frame is not None and frame > prev_frame + 1:
                add_outside_box(track_elem, min(prev_frame + 1, stop_frame), prev_box)
            ET.SubElement(
                track_elem,
                "box",
                {
                    "frame": str(frame),
                    "keyframe": "1",
                    "outside": "0",
                    "occluded": "0",
                    "xtl": f"{box[0]:.2f}",
                    "ytl": f"{box[1]:.2f}",
                    "xbr": f"{box[2]:.2f}",
                    "ybr": f"{box[3]:.2f}",
                    "z_order": "0",
                },
            )
            prev_frame = frame
            prev_box = box
        if prev_frame is not None and prev_box is not None:
            add_outside_box(track_elem, min(prev_frame + 1, stop_frame), prev_box)
        created += 1
    return created, skipped


def write_xml(root, out_path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def upload_xml(task_id, xml_path, args):
    zip_path = xml_path.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_path, "annotations.xml")
    run_cmd(
        cvat_base_cmd(args)
        + ["task", "import-dataset", str(task_id), str(zip_path), "--format", "CVAT 1.1"],
        env={"PASS": args.cvat_pass},
    )


def process_chunk(chunk_name, task_id, model, args):
    print(f"\n[CHUNK] {chunk_name} task={task_id}")
    tracking_data, processed_frames = run_tracking(chunk_name, model, args)
    clean_data = filter_short_tracks(tracking_data, fps_target=30)

    json_path = JSON_DIR / f"{chunk_name}_lightglue_tracks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False)

    root = export_task_template(task_id, chunk_name, args)
    clear_existing_annotations(root)
    n_tracks, n_skipped = build_track_elements(root, clean_data)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xml_path = XML_DIR / f"{chunk_name}_task{task_id}_{ts}.xml"
    write_xml(root, xml_path)
    if args.upload:
        upload_xml(task_id, xml_path, args)

    print(f"[DONE] {chunk_name} frames={processed_frames} tracks={n_tracks} uploaded={args.upload}")
    return {
        "chunk_name": chunk_name,
        "task_id": task_id,
        "processed_frames": processed_frames,
        "raw_sightings": len(tracking_data),
        "clean_sightings": len(clean_data),
        "unique_track_ids": len({row["id"] for row in clean_data}),
        "xml_tracks_created": n_tracks,
        "xml_tracks_skipped_class_out_of_range": n_skipped,
        "json_path": str(json_path),
        "xml_path": str(xml_path),
        "uploaded": args.upload,
    }


def main():
    parser = argparse.ArgumentParser(description="Roda tracker nos chunks balanceados e importa no CVAT.")
    parser.add_argument("--balanced-root", default=str(BALANCED_ROOT))
    parser.add_argument("--results-root", default=str(BALANCED_RESULTS_ROOT))
    parser.add_argument("--task-prefix", default="cam1_plate_balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--reset-gap", type=int, default=1)
    parser.add_argument("--project-id", type=int, default=CVAT_PROJECT_ID)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--only-chunk", nargs="*", default=None)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--cvat-host", default=CVAT_HOST)
    parser.add_argument("--cvat-port", default=CVAT_PORT)
    parser.add_argument("--cvat-org", default=CVAT_ORG)
    parser.add_argument("--cvat-user", default=CVAT_USER)
    parser.add_argument("--cvat-pass", default=CVAT_PASS)
    args = parser.parse_args()

    ensure_dirs()
    task_map = get_task_map(args)
    chunks = args.only_chunk if args.only_chunk else sorted(task_map)
    model = YOLO(args.model)
    summaries = []
    for chunk_name in chunks:
        if chunk_name not in task_map:
            print(f"[SKIP] Sem task para {chunk_name}")
            continue
        summaries.append(process_chunk(chunk_name, task_map[chunk_name], model, args))

    summary_path = OUTPUT_ROOT / f"run_summary_{int(time.time())}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
