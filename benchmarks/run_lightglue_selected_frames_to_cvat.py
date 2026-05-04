import argparse
import json
import math
import os
import re
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


VIDEOS = [
    "GX010084",
    "GX010083",
    "GX010069",
    "GX010076",
    "GX010086",
    "GX010067",
    "GX010085",
]

ROOT = Path("/home/servidor/ArtigoLightglue")
MODEL_PATH = ROOT / "models/detector_placa.pt"
SELECTED_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_plate_selected_frames")
SELECTION_RESULTS_ROOT = ROOT / "results/plate_frame_selection"
OUTPUT_ROOT = ROOT / "results/cvat_lightglue_selected_frames"
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
    suffix = args.task_suffix
    for task in tasks:
        if task.get("project_id") != args.project_id:
            continue
        name = task.get("name", "")
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        mapping[stem] = int(task["id"])
    return mapping


def parse_frame_number(path, stem):
    match = re.search(rf"{re.escape(stem)}_frame_(\d+)\.jpg$", path.name)
    if not match:
        raise ValueError(f"Nome de frame inesperado: {path}")
    return int(match.group(1))


def load_manifest(stem, frame_dir):
    selection_json = SELECTION_RESULTS_ROOT / f"{stem}_selection.json"
    if selection_json.exists():
        with open(selection_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for row in data["manifest"]:
            path = Path(row["selected_path"])
            if not path.exists():
                path = frame_dir / row["file_name"]
            items.append(
                {
                    "task_frame": int(row["task_frame"]),
                    "original_frame": int(row["original_frame"]),
                    "path": path,
                    "file_name": row["file_name"],
                }
            )
        return sorted(items, key=lambda x: x["task_frame"])

    paths = sorted(frame_dir.glob(f"{stem}_frame_*.jpg"), key=lambda p: parse_frame_number(p, stem))
    return [
        {
            "task_frame": i,
            "original_frame": parse_frame_number(path, stem),
            "path": path,
            "file_name": path.name,
        }
        for i, path in enumerate(paths)
    ]


def new_tracker(device, next_id):
    tracker = LightGlueTracker(device=device, accept_th=1.2, motion_weight=0.3)
    tracker.next_id = next_id
    return tracker


def run_lightglue_tracking_on_frames(stem, frame_dir, model, args):
    items = load_manifest(stem, frame_dir)
    if not items:
        return [], 0

    tracker = new_tracker(args.device, 1)
    next_id = 1
    prev_original_frame = None
    tracking_data = []

    pbar = tqdm(items, desc=stem)
    for item in pbar:
        original_frame = int(item["original_frame"])
        task_frame = int(item["task_frame"])

        if prev_original_frame is not None and original_frame - prev_original_frame > args.reset_gap:
            next_id = tracker.next_id
            tracker = new_tracker(args.device, next_id)

        frame = cv2.imread(str(item["path"]))
        if frame is None:
            print(f"[WARN] Nao consegui ler: {item['path']}")
            prev_original_frame = original_frame
            continue

        det_res = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            classes=list(range(11)),
            device=args.device,
        )[0]

        det_bboxes = []
        det_classes = []
        if det_res.boxes is not None:
            det_bboxes = det_res.boxes.xyxy.cpu().numpy()
            det_classes = det_res.boxes.cls.cpu().numpy().astype(int)

        active_tracks = tracker.update(
            det_bboxes,
            frame,
            det_classes,
            original_frame,
            stride=1,
        )
        for trk in active_tracks:
            tracking_data.append(
                {
                    "frame": task_frame,
                    "original_frame": original_frame,
                    "id": int(trk["id"]),
                    "box": [float(x) for x in trk["bbox"]],
                    "cls": int(trk["cls"]),
                    "file_name": item["file_name"],
                }
            )

        prev_original_frame = original_frame

    pbar.close()
    return tracking_data, len(items)


def filter_short_tracks(tracking_data, fps_target=30):
    id_counts = defaultdict(int)
    for d in tracking_data:
        id_counts[d["id"]] += 1
    min_presence = max(1, math.ceil(0.3 * fps_target))
    return [d for d in tracking_data if id_counts[d["id"]] >= min_presence]


def export_task_template(task_id, stem, args):
    out_zip = TEMPLATE_DIR / f"{stem}_task{task_id}_template.zip"
    if out_zip.exists():
        out_zip.unlink()
    cmd = cvat_base_cmd(args) + [
        "task",
        "export-dataset",
        str(task_id),
        str(out_zip),
        "--format",
        "CVAT for images 1.1",
    ]
    run_cmd(cmd, env={"PASS": args.cvat_pass})
    with zipfile.ZipFile(out_zip) as zf:
        xml_data = zf.read("annotations.xml")
    return ET.fromstring(xml_data)


def clear_existing_annotations(root):
    for image_elem in root.findall("image"):
        for child in list(image_elem):
            image_elem.remove(child)
    for track_elem in list(root.findall("track")):
        root.remove(track_elem)


def extract_label_names(root):
    labels = []
    label_nodes = root.findall("./meta/task/labels/label/name")
    if not label_nodes:
        label_nodes = root.findall("./meta/job/labels/label/name")
    for node in label_nodes:
        labels.append((node.text or "").strip())
    return labels


def to_xml_float(val):
    return f"{val:.2f}"


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
    for d in clean_data:
        grouped[d["id"]].append(d)
    for tid in grouped:
        grouped[tid].sort(key=lambda x: x["frame"])
    return grouped


def majority_class(track_obs):
    cls_count = defaultdict(int)
    for obs in track_obs:
        cls_count[obs["cls"]] += 1
    return max(cls_count.items(), key=lambda kv: kv[1])[0]


def build_track_elements(root, clean_data):
    images = root.findall("image")
    if not images:
        raise RuntimeError("Template XML sem nós <image>.")

    width = int(images[0].attrib.get("width", "1920"))
    height = int(images[0].attrib.get("height", "1080"))
    stop_frame = max(int(img.attrib["id"]) for img in images)
    labels = extract_label_names(root)
    if not labels:
        raise RuntimeError("Não foi possível ler labels do template XML.")

    grouped = group_tracks(clean_data)
    created = 0
    skipped = 0
    for track_id in sorted(grouped.keys()):
        obs = grouped[track_id]
        cls_id = majority_class(obs)
        if cls_id < 0 or cls_id >= len(labels):
            skipped += 1
            continue
        label_name = labels[cls_id]
        track_elem = ET.SubElement(
            root,
            "track",
            {"id": str(track_id), "label": label_name, "source": "auto"},
        )

        prev_frame = None
        prev_box = None
        for item in obs:
            frame = int(item["frame"])
            box = clamp_box(item["box"], width, height)

            if prev_frame is not None and frame > prev_frame + 1:
                off_frame = min(prev_frame + 1, stop_frame)
                ET.SubElement(
                    track_elem,
                    "box",
                    {
                        "frame": str(off_frame),
                        "keyframe": "1",
                        "outside": "1",
                        "occluded": "0",
                        "xtl": to_xml_float(prev_box[0]),
                        "ytl": to_xml_float(prev_box[1]),
                        "xbr": to_xml_float(prev_box[2]),
                        "ybr": to_xml_float(prev_box[3]),
                        "z_order": "0",
                    },
                )

            ET.SubElement(
                track_elem,
                "box",
                {
                    "frame": str(frame),
                    "keyframe": "1",
                    "outside": "0",
                    "occluded": "0",
                    "xtl": to_xml_float(box[0]),
                    "ytl": to_xml_float(box[1]),
                    "xbr": to_xml_float(box[2]),
                    "ybr": to_xml_float(box[3]),
                    "z_order": "0",
                },
            )
            prev_frame = frame
            prev_box = box

        if prev_frame is not None and prev_box is not None:
            off_frame = min(prev_frame + 1, stop_frame)
            if off_frame >= prev_frame:
                ET.SubElement(
                    track_elem,
                    "box",
                    {
                        "frame": str(off_frame),
                        "keyframe": "1",
                        "outside": "1",
                        "occluded": "0",
                        "xtl": to_xml_float(prev_box[0]),
                        "ytl": to_xml_float(prev_box[1]),
                        "xbr": to_xml_float(prev_box[2]),
                        "ybr": to_xml_float(prev_box[3]),
                        "z_order": "0",
                    },
                )

        created += 1
    return created, skipped


def write_xml(root, out_path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def upload_xml(task_id, xml_path, args):
    cmd = cvat_base_cmd(args) + [
        "task",
        "import-dataset",
        str(task_id),
        str(xml_path),
        "--format",
        "CVAT 1.1",
    ]
    run_cmd(cmd, env={"PASS": args.cvat_pass})


def process_stem(stem, task_id, model, args):
    frame_dir = SELECTED_ROOT / stem
    if not frame_dir.exists():
        print(f"[SKIP] Pasta nao encontrada: {frame_dir}")
        return None

    print(f"\n[VIDEO] {stem}")
    print(f"[FRAMES] {frame_dir}")
    print(f"[TASK] {task_id}")

    tracking_data, processed_frames = run_lightglue_tracking_on_frames(
        stem,
        frame_dir,
        model,
        args,
    )
    clean_data = filter_short_tracks(tracking_data, fps_target=30)

    json_path = JSON_DIR / f"{stem}_selected_lightglue_tracks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False)

    root = export_task_template(task_id, stem, args)
    clear_existing_annotations(root)
    n_tracks, n_skipped = build_track_elements(root, clean_data)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xml_path = XML_DIR / f"{stem}_selected_annotations_task{task_id}_{ts}.xml"
    write_xml(root, xml_path)

    if args.upload:
        upload_xml(task_id, xml_path, args)

    summary = {
        "video": stem,
        "task_id": task_id,
        "processed_frames": processed_frames,
        "raw_sightings": len(tracking_data),
        "clean_sightings": len(clean_data),
        "unique_track_ids": len({d["id"] for d in clean_data}),
        "xml_tracks_created": n_tracks,
        "xml_tracks_skipped_class_out_of_range": n_skipped,
        "json_path": str(json_path),
        "xml_path": str(xml_path),
        "uploaded": args.upload,
    }
    print(f"[DONE] {stem} | tracks={n_tracks} | sightings={len(clean_data)}")
    return summary


def main():
    global SELECTED_ROOT

    parser = argparse.ArgumentParser(
        description="Roda LightGlue nos frames filtrados e gera/importa XML CVAT."
    )
    parser.add_argument("--selected-root", default=str(SELECTED_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--reset-gap", type=int, default=1)
    parser.add_argument("--task-suffix", default="_plate_frames")
    parser.add_argument("--project-id", type=int, default=CVAT_PROJECT_ID)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--only-stem", nargs="*", default=None)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--cvat-host", default=CVAT_HOST)
    parser.add_argument("--cvat-port", default=CVAT_PORT)
    parser.add_argument("--cvat-org", default=CVAT_ORG)
    parser.add_argument("--cvat-user", default=CVAT_USER)
    parser.add_argument("--cvat-pass", default=CVAT_PASS)
    args = parser.parse_args()

    SELECTED_ROOT = Path(args.selected_root)

    ensure_dirs()
    task_map = get_task_map(args)
    if not task_map:
        raise RuntimeError(f"Nenhuma task *{args.task_suffix} encontrada no projeto {args.project_id}.")

    stems = args.only_stem if args.only_stem else VIDEOS
    model = YOLO(args.model)
    all_stats = []

    for stem in stems:
        if stem not in task_map:
            print(f"[SKIP] Sem task para {stem} (esperado: {stem}{args.task_suffix}).")
            continue
        stats = process_stem(stem, task_map[stem], model, args)
        if stats:
            all_stats.append(stats)

    summary_path = OUTPUT_ROOT / f"run_summary_{int(time.time())}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print("\n=== FINALIZADO ===")
    print(f"Resumo: {summary_path}")
    for row in all_stats:
        print(
            f"- {row['video']} task={row['task_id']} "
            f"tracks={row['xml_tracks_created']} uploaded={row['uploaded']}"
        )


if __name__ == "__main__":
    main()
