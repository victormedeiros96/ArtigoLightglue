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

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.lightglue_tracker import LightGlueTracker


# VIDEOS = [
#     "/home/servidor/ArtigoLightglue/proxies/GX010084.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010083.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010069.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010076.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010086.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010067.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010080.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010081.MP4",
#     "/home/servidor/ArtigoLightglue/proxies/GX010085.MP4",
# ]


VIDEOS = [
    "/mnt/hd2/tasks_stag_go_pro_dez/cam1/GX010084.MP4",
    "/mnt/hd2/tasks_stag_go_pro_dez/cam1/GX010083.MP4",
    "/mnt/hd2/tasks_stag_go_pro_dez/cam1/GX010069.MP4",
    "/mnt/hd2/tasks_stag_go_pro_dez/cam1/GX010076.MP4",
    "/mnt/hd2/tasks_stag_go_pro_dez/cam1/GX010086.MP4",
    "/mnt/hd2/tasks_stag_go_pro_dez/cam1/GX010067.MP4",
    "/mnt/hd2/tasks_stag_go_pro_dez/cam1/GX010085.MP4",
]


ROOT = "/home/servidor/ArtigoLightglue"
MODEL_PATH = f"{ROOT}/models/detector_placa.pt"
OUTPUT_ROOT = f"{ROOT}/results/cvat_lightglue_tracking"
XML_DIR = f"{OUTPUT_ROOT}/xmls"
JSON_DIR = f"{OUTPUT_ROOT}/jsons"
TEMPLATE_DIR = f"{OUTPUT_ROOT}/templates"
LOG_DIR = f"{OUTPUT_ROOT}/logs"

CVAT_HOST = "http://192.168.18.140"
CVAT_PORT = "8080"
CVAT_ORG = "RDT3"
CVAT_PROJECT_ID = 79
CVAT_USER = "superrdt"
CVAT_PASS = "superpwdrdt"


def ensure_dirs():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(XML_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


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


def cvat_base_cmd():
    return [
        "cvat-cli",
        "--auth",
        CVAT_USER,
        "--server-host",
        CVAT_HOST,
        "--server-port",
        CVAT_PORT,
        "--organization",
        CVAT_ORG,
    ]


def get_task_map():
    cmd = cvat_base_cmd() + ["task", "ls", "--json"]
    output = run_cmd(cmd, env={"PASS": CVAT_PASS})
    tasks = json.loads(output)
    mapping = {}
    for task in tasks:
        if task.get("project_id") != CVAT_PROJECT_ID:
            continue
        name = task.get("name", "")
        if not name.endswith("_frames"):
            continue
        stem = name[: -len("_frames")]
        mapping[stem] = int(task["id"])
    return mapping


def resolve_video_path(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if stem == "GX010081":
        recovered = f"{ROOT}/proxies/GX010081_recovered.mp4"
        if os.path.exists(recovered):
            return recovered
    return video_path


def run_lightglue_tracking(video_path, model, device, max_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max_frames
    stop_frame = min(total_frames, max_frames) - 1
    if stop_frame < 0:
        cap.release()
        return [], 0, -1

    tracker = LightGlueTracker(
        device=device,
        accept_th=1.2,
        motion_weight=0.3,
    )

    tracking_data = []
    frame_idx = 0
    pbar = tqdm(total=stop_frame + 1, desc=os.path.basename(video_path))
    while cap.isOpened() and frame_idx <= stop_frame:
        ok, frame = cap.read()
        if not ok:
            break

        det_res = model.predict(
            frame,
            imgsz=1280,
            verbose=False,
            classes=list(range(11)),
            device=device,
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
            frame_idx,
            stride=1,
        )
        for trk in active_tracks:
            tracking_data.append(
                {
                    "frame": int(frame_idx),
                    "id": int(trk["id"]),
                    "box": [float(x) for x in trk["bbox"]],
                    "cls": int(trk["cls"]),
                }
            )

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return tracking_data, frame_idx, stop_frame


def filter_short_tracks(tracking_data, fps_target=30):
    id_counts = defaultdict(int)
    for d in tracking_data:
        id_counts[d["id"]] += 1
    min_presence = max(1, math.ceil(0.3 * fps_target))
    return [d for d in tracking_data if id_counts[d["id"]] >= min_presence]


def export_task_template(task_id, stem):
    out_zip = f"{TEMPLATE_DIR}/{stem}_task{task_id}_template.zip"
    if os.path.exists(out_zip):
        os.remove(out_zip)
    cmd = cvat_base_cmd() + [
        "task",
        "export-dataset",
        str(task_id),
        out_zip,
        "--format",
        "CVAT for images 1.1",
    ]
    run_cmd(cmd, env={"PASS": CVAT_PASS})
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
            {
                "id": str(track_id),
                "label": label_name,
                "source": "auto",
            },
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


def upload_xml(task_id, xml_path):
    cmd = cvat_base_cmd() + [
        "task",
        "import-dataset",
        str(task_id),
        xml_path,
        "--format",
        "CVAT 1.1",
    ]
    run_cmd(cmd, env={"PASS": CVAT_PASS})


def process_video(video_path, task_id, model, device, max_frames, upload):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    source_video = resolve_video_path(video_path)
    print(f"\n[VIDEO] {stem}")
    print(f"[SOURCE] {source_video}")
    print(f"[TASK] {task_id}")

    tracking_data, processed_frames, stop_frame = run_lightglue_tracking(
        source_video,
        model=model,
        device=device,
        max_frames=max_frames,
    )
    clean_data = filter_short_tracks(tracking_data, fps_target=30)

    json_path = f"{JSON_DIR}/{stem}_lightglue_tracks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False)

    root = export_task_template(task_id, stem)
    clear_existing_annotations(root)
    n_tracks, n_skipped = build_track_elements(root, clean_data)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xml_path = f"{XML_DIR}/{stem}_annotations_task{task_id}_{ts}.xml"
    write_xml(root, xml_path)

    if upload:
        upload_xml(task_id, xml_path)

    unique_ids = len({d["id"] for d in clean_data})
    summary = {
        "video": stem,
        "task_id": task_id,
        "processed_frames": processed_frames,
        "stop_frame": stop_frame,
        "raw_sightings": len(tracking_data),
        "clean_sightings": len(clean_data),
        "unique_track_ids": unique_ids,
        "xml_tracks_created": n_tracks,
        "xml_tracks_skipped_class_out_of_range": n_skipped,
        "json_path": json_path,
        "xml_path": xml_path,
        "uploaded": upload,
    }
    print(f"[DONE] {stem} | tracks={n_tracks} | sightings={len(clean_data)}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Roda LightGlue tracking e gera/importa XML CVAT 1.1 por vídeo."
    )
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, etc.")
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--upload", action="store_true", help="Importa XML nas tasks CVAT.")
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Lista de vídeos para processar. Se vazio, usa a lista padrão.",
    )
    parser.add_argument(
        "--only-stem",
        nargs="*",
        default=None,
        help="Filtra pelos stems dos vídeos (ex: GX010084 GX010085).",
    )
    args = parser.parse_args()

    ensure_dirs()
    task_map = get_task_map()
    if not task_map:
        raise RuntimeError("Nenhuma task *_frames encontrada no projeto 79.")

    videos = args.videos if args.videos else VIDEOS
    if args.only_stem:
        wanted = set(args.only_stem)
        videos = [v for v in videos if os.path.splitext(os.path.basename(v))[0] in wanted]

    model = YOLO(MODEL_PATH)
    run_started = int(time.time())
    all_stats = []

    for video_path in videos:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        if stem not in task_map:
            print(f"[SKIP] Sem task para {stem} (esperado: {stem}_frames).")
            continue
        if not os.path.exists(video_path):
            print(f"[SKIP] Vídeo não encontrado: {video_path}")
            continue

        stats = process_video(
            video_path=video_path,
            task_id=task_map[stem],
            model=model,
            device=args.device,
            max_frames=args.max_frames,
            upload=args.upload,
        )
        all_stats.append(stats)

    summary_path = f"{OUTPUT_ROOT}/run_summary_{run_started}.json"
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
