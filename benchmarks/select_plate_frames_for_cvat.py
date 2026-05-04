import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


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
FRAMES_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_frames")
SELECTED_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_plate_selected_frames")
RESULTS_ROOT = ROOT / "results/plate_frame_selection"
DEFAULT_CLASSES = list(range(11))


def ensure_dirs():
    SELECTED_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def parse_classes(value):
    if value.strip().lower() in {"all", "*"}:
        return None
    classes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        classes.append(int(part))
    return classes


def parse_frame_number(path, stem):
    match = re.search(rf"{re.escape(stem)}_frame_(\d+)\.jpg$", path.name)
    if not match:
        raise ValueError(f"Nome de frame inesperado: {path}")
    return int(match.group(1))


def list_frames(stem):
    frame_dir = FRAMES_ROOT / stem
    if not frame_dir.exists():
        raise FileNotFoundError(f"Pasta de frames nao encontrada: {frame_dir}")
    frames = sorted(
        frame_dir.glob(f"{stem}_frame_*.jpg"),
        key=lambda p: parse_frame_number(p, stem),
    )
    if not frames:
        raise RuntimeError(f"Nenhum frame encontrado em: {frame_dir}")
    return frames


def clean_selected_dir(out_dir, stem):
    if out_dir.exists():
        for path in out_dir.glob(f"{stem}_frame_*.jpg"):
            path.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)


def make_symlinks(stem, selected_paths, force):
    out_dir = SELECTED_ROOT / stem
    if force:
        clean_selected_dir(out_dir, stem)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    for src in selected_paths:
        dst = out_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(src, dst)
        linked += 1
    return out_dir, linked


def expand_with_window(detected_frames, existing_frames, window):
    existing = set(existing_frames)
    selected = set()
    for frame_idx in detected_frames:
        start = max(min(existing), frame_idx - window)
        stop = min(max(existing), frame_idx + window)
        for idx in range(start, stop + 1):
            if idx in existing:
                selected.add(idx)
    return sorted(selected)


def compact_ranges(values):
    if not values:
        return []
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = prev = value
    ranges.append([start, prev])
    return ranges


def parse_devices(value):
    devices = []
    for part in value.split(","):
        part = part.strip()
        if part:
            devices.append(part)
    if not devices:
        raise ValueError("--multi-gpu-devices nao pode ficar vazio")
    return devices


def balance_stems_by_frame_count(stems, devices, max_frames):
    buckets = [{"device": device, "stems": [], "frames": 0} for device in devices]
    stem_sizes = []
    for stem in stems:
        frame_count = len(list_frames(stem))
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        stem_sizes.append((stem, frame_count))

    for stem, frame_count in sorted(stem_sizes, key=lambda item: item[1], reverse=True):
        bucket = min(buckets, key=lambda item: item["frames"])
        bucket["stems"].append(stem)
        bucket["frames"] += frame_count

    return [bucket for bucket in buckets if bucket["stems"]]


def run_multi_gpu(args, stems):
    devices = parse_devices(args.multi_gpu_devices)
    buckets = balance_stems_by_frame_count(stems, devices, args.max_frames)
    script_path = Path(__file__).resolve()
    processes = []

    print("[MULTI-GPU] Distribuicao:")
    for bucket in buckets:
        print(
            f"  GPU {bucket['device']}: {', '.join(bucket['stems'])} "
            f"({bucket['frames']} frames)"
        )

    for bucket in buckets:
        cmd = [
            sys.executable,
            str(script_path),
            "--frames-root",
            str(args.frames_root),
            "--selected-root",
            str(args.selected_root),
            "--results-root",
            str(args.results_root),
            "--model",
            str(args.model),
            "--device",
            "auto",
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--batch",
            str(args.batch),
            "--loader-workers",
            str(args.loader_workers),
            "--torch-loader-workers",
            str(args.torch_loader_workers),
            "--prefetch-batches",
            str(args.prefetch_batches),
            "--window",
            str(args.window),
            "--classes",
            str(args.classes),
            "--only-stem",
            *bucket["stems"],
        ]
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.force:
            cmd.append("--force")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(bucket["device"])
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[MULTI-GPU] Start GPU {bucket['device']}: {' '.join(bucket['stems'])}")
        processes.append((bucket, subprocess.Popen(cmd, env=env)))

    failed = []
    for bucket, process in processes:
        returncode = process.wait()
        if returncode != 0:
            failed.append((bucket, returncode))

    if failed:
        for bucket, returncode in failed:
            print(
                f"[ERROR] GPU {bucket['device']} stems={bucket['stems']} "
                f"retornou codigo {returncode}"
            )
        raise SystemExit(1)

    all_stats = []
    for stem in stems:
        selection_path = Path(args.results_root) / f"{stem}_selection.json"
        with open(selection_path, "r", encoding="utf-8") as f:
            all_stats.append(json.load(f)["summary"])

    summary_path = Path(args.results_root) / f"selection_summary_multigpu_{int(time.time())}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n[MULTI-GPU SUMMARY] {summary_path}")


def read_frame(path):
    image = cv2.imread(str(path))
    if image is None:
        print(f"[WARN] Nao consegui ler: {path}")
    return path, image


class FrameDataset:
    def __init__(self, frames):
        self.frames = [str(path) for path in frames]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = Path(self.frames[index])
        return read_frame(path)


def collate_loaded_frames(items):
    batch_paths = [path for path, _ in items]
    return batch_paths, items


def iter_torch_loaded_batches(frames, batch_size, torch_loader_workers):
    from torch.utils.data import DataLoader

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": torch_loader_workers,
        "collate_fn": collate_loaded_frames,
        "pin_memory": False,
        "persistent_workers": torch_loader_workers > 0,
    }
    if torch_loader_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(FrameDataset(frames), **loader_kwargs)
    for batch_paths, loaded in loader:
        yield batch_paths, loaded


def iter_loaded_batches(
    frames,
    batch_size,
    loader_workers,
    prefetch_batches,
    torch_loader_workers,
):
    if torch_loader_workers > 0:
        yield from iter_torch_loaded_batches(frames, batch_size, torch_loader_workers)
        return

    if loader_workers <= 0:
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            loaded = [read_frame(path) for path in batch]
            yield batch, loaded
        return

    if prefetch_batches <= 0:
        with ThreadPoolExecutor(max_workers=loader_workers) as executor:
            for start in range(0, len(frames), batch_size):
                batch = frames[start : start + batch_size]
                loaded = list(executor.map(read_frame, batch))
                yield batch, loaded
        return

    loaded_queue = queue.Queue(maxsize=max(1, prefetch_batches))
    sentinel = object()

    def producer():
        try:
            with ThreadPoolExecutor(max_workers=loader_workers) as executor:
                for start in range(0, len(frames), batch_size):
                    batch = frames[start : start + batch_size]
                    loaded = list(executor.map(read_frame, batch))
                    loaded_queue.put((batch, loaded))
        except BaseException as exc:
            loaded_queue.put(exc)
        finally:
            loaded_queue.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        item = loaded_queue.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item

    thread.join()


def infer_and_select(stem, model, args, classes):
    frames = list_frames(stem)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    frame_numbers = [parse_frame_number(p, stem) for p in frames]
    frame_by_number = dict(zip(frame_numbers, frames))

    detections = {}
    detected_frame_numbers = []
    pbar = tqdm(total=len(frames), desc=f"{stem} infer")
    for batch, loaded in iter_loaded_batches(
        frames,
        args.batch,
        args.loader_workers,
        args.prefetch_batches,
        args.torch_loader_workers,
    ):
        batch_images = []
        batch_paths = []
        for path, image in loaded:
            if image is None:
                continue
            batch_images.append(image)
            batch_paths.append(path)

        if not batch_images:
            pbar.update(len(batch))
            continue

        predict_kwargs = {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "classes": classes,
            "verbose": False,
        }
        if args.device != "auto":
            predict_kwargs["device"] = args.device
        results = model.predict(batch_images, **predict_kwargs)

        for path, result in zip(batch_paths, results):
            frame_number = parse_frame_number(path, stem)
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            frame_dets = []
            for box, conf, cls_id in zip(xyxy, confs, clss):
                frame_dets.append(
                    {
                        "cls": int(cls_id),
                        "conf": float(conf),
                        "box": [float(x) for x in box],
                    }
                )
            if frame_dets:
                detections[str(frame_number)] = frame_dets
                detected_frame_numbers.append(frame_number)

        pbar.update(len(batch))
    pbar.close()

    selected_numbers = expand_with_window(
        sorted(set(detected_frame_numbers)),
        frame_numbers,
        args.window,
    )
    selected_paths = [frame_by_number[n] for n in selected_numbers]
    out_dir, linked = make_symlinks(stem, selected_paths, args.force)

    manifest = []
    for task_frame, frame_number in enumerate(selected_numbers):
        src = frame_by_number[frame_number]
        manifest.append(
            {
                "task_frame": task_frame,
                "original_frame": frame_number,
                "file_name": src.name,
                "source_path": str(src),
                "selected_path": str(out_dir / src.name),
                "has_detection": str(frame_number) in detections,
            }
        )

    summary = {
        "video": stem,
        "source_dir": str(FRAMES_ROOT / stem),
        "selected_dir": str(out_dir),
        "total_frames": len(frames),
        "detected_frames": len(set(detected_frame_numbers)),
        "selected_frames": len(selected_numbers),
        "window": args.window,
        "classes": classes if classes is not None else "all",
        "conf": args.conf,
        "imgsz": args.imgsz,
        "linked_now": linked,
        "selected_ranges": compact_ranges(selected_numbers),
        "created_at": int(time.time()),
    }

    out_json = RESULTS_ROOT / f"{stem}_selection.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "detections": detections,
                "manifest": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[DONE] {stem} detected={summary['detected_frames']} "
        f"selected={summary['selected_frames']} linked={linked}"
    )
    return summary


def main():
    global FRAMES_ROOT, SELECTED_ROOT, RESULTS_ROOT

    parser = argparse.ArgumentParser(
        description=(
            "Roda detector nos frames extraidos, seleciona frames com deteccao "
            "mais janela temporal e cria pastas de symlinks para CVAT."
        )
    )
    parser.add_argument("--frames-root", default=str(FRAMES_ROOT))
    parser.add_argument("--selected-root", default=str(SELECTED_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help=(
            "Numero de workers para carregar JPGs em paralelo antes da inferencia. "
            "0 mantem leitura sequencial."
        ),
    )
    parser.add_argument(
        "--torch-loader-workers",
        type=int,
        default=0,
        help=(
            "Usa torch.utils.data.DataLoader com este numero de workers para "
            "carregar JPGs. Quando >0, substitui --loader-workers."
        ),
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=0,
        help=(
            "Numero maximo de batches carregados esperando a GPU. "
            "0 desativa fila e apenas paraleliza leitura dentro do batch."
        ),
    )
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--classes",
        default=",".join(str(c) for c in DEFAULT_CLASSES),
        help="Classes separadas por virgula. Use 'all' para nao filtrar.",
    )
    parser.add_argument(
        "--multi-gpu-devices",
        default=None,
        help=(
            "Lista de GPUs fisicas separadas por virgula para distribuir videos "
            "em subprocessos. Exemplo: 0,1,2,3."
        ),
    )
    parser.add_argument("--only-stem", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", help="Recria symlinks existentes.")
    args = parser.parse_args()

    FRAMES_ROOT = Path(args.frames_root)
    SELECTED_ROOT = Path(args.selected_root)
    RESULTS_ROOT = Path(args.results_root)
    ensure_dirs()

    classes = parse_classes(args.classes)
    stems = args.only_stem if args.only_stem else VIDEOS
    if args.multi_gpu_devices:
        run_multi_gpu(args, stems)
        return

    model = YOLO(args.model)

    all_stats = []
    for stem in stems:
        all_stats.append(infer_and_select(stem, model, args, classes))

    summary_path = RESULTS_ROOT / f"selection_summary_{int(time.time())}_{os.getpid()}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
