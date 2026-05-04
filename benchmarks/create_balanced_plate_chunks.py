import argparse
import json
import os
import shutil
import time
from pathlib import Path


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
FRAMES_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_frames")
SELECTION_RESULTS_ROOT = ROOT / "results/plate_frame_selection"
BALANCED_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_plate_balanced_frames")
BALANCED_RESULTS_ROOT = ROOT / "results/plate_balanced_chunks"


def load_selection(stem, selection_root):
    path = selection_root / f"{stem}_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"Selection JSON nao encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def connected_detection_intervals(detected_frames, total_frames, window):
    intervals = []
    for frame in sorted(set(detected_frames)):
        start = max(0, frame - window)
        stop = min(total_frames - 1, frame + window)
        if not intervals or start > intervals[-1][1] + 1:
            intervals.append([start, stop])
        else:
            intervals[-1][1] = max(intervals[-1][1], stop)
    return intervals


def interval_size(interval):
    return int(interval["stop"]) - int(interval["start"]) + 1


def build_intervals(stems, selection_root, frames_root, window):
    intervals = []
    for stem in stems:
        data = load_selection(stem, selection_root)
        total_frames = int(data["summary"]["total_frames"])
        detected = [int(k) for k in data["detections"].keys()]
        for start, stop in connected_detection_intervals(detected, total_frames, window):
            intervals.append(
                {
                    "video": stem,
                    "start": start,
                    "stop": stop,
                    "size": stop - start + 1,
                    "source_dir": str(frames_root / stem),
                }
            )
    return intervals


def build_intervals_by_video(stems, selection_root, frames_root, window):
    return {
        stem: build_intervals([stem], selection_root, frames_root, window)
        for stem in stems
    }


def split_chunks(intervals, min_task_size):
    chunks = []
    current = []
    current_size = 0
    for interval in intervals:
        current.append(interval)
        current_size += interval_size(interval)
        if current_size >= min_task_size:
            chunks.append(current)
            current = []
            current_size = 0
    if current:
        chunks.append(current)
    return chunks


def clean_root(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def source_frame_path(frames_root, stem, frame_number):
    return frames_root / stem / f"{stem}_frame_{frame_number:06d}.jpg"


def create_link(src, dst, link_type):
    if dst.exists():
        dst.unlink()
    if link_type == "symlink":
        os.symlink(src, dst)
    elif link_type == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"link_type invalido: {link_type}")


def create_chunk_links(chunks, args):
    rows = []
    for chunk_index, item in enumerate(chunks, start=1):
        if isinstance(item, tuple):
            video_stem, local_index, intervals = item
            chunk_name = f"{args.task_prefix}_{video_stem}_{local_index:04d}"
        else:
            intervals = item
            chunk_name = f"{args.task_prefix}_{chunk_index:04d}"
        chunk_dir = Path(args.balanced_root) / chunk_name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        task_frame = 0

        for interval in intervals:
            stem = interval["video"]
            for original_frame in range(int(interval["start"]), int(interval["stop"]) + 1):
                src = source_frame_path(Path(args.frames_root), stem, original_frame)
                if not src.exists():
                    raise FileNotFoundError(f"Frame nao encontrado: {src}")
                link_name = f"{chunk_name}_frame_{task_frame:06d}.jpg"
                dst = chunk_dir / link_name
                create_link(src, dst, args.link_type)
                manifest.append(
                    {
                        "task_frame": task_frame,
                        "video": stem,
                        "original_frame": original_frame,
                        "file_name": link_name,
                        "source_file_name": src.name,
                        "source_path": str(src),
                        "selected_path": str(dst),
                    }
                )
                task_frame += 1

        chunk_json = Path(args.results_root) / f"{chunk_name}.json"
        summary = {
            "chunk_name": chunk_name,
            "chunk_dir": str(chunk_dir),
            "frames": len(manifest),
            "intervals": intervals,
            "created_at": int(time.time()),
        }
        with open(chunk_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "manifest": manifest}, f, indent=2, ensure_ascii=False)
        rows.append(summary | {"manifest_path": str(chunk_json)})
        print(f"[CHUNK] {chunk_name} frames={len(manifest)} intervals={len(intervals)}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Cria chunks balanceados por intervalos conectados de deteccao."
    )
    parser.add_argument("--frames-root", default=str(FRAMES_ROOT))
    parser.add_argument("--selection-root", default=str(SELECTION_RESULTS_ROOT))
    parser.add_argument("--balanced-root", default=str(BALANCED_ROOT))
    parser.add_argument("--results-root", default=str(BALANCED_RESULTS_ROOT))
    parser.add_argument("--task-prefix", default="cam1_plate_balanced")
    parser.add_argument("--min-task-size", type=int, default=4000)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--only-stem", nargs="*", default=None)
    parser.add_argument(
        "--split-per-video",
        action="store_true",
        help="Nao mistura videos no mesmo chunk; reinicia o balanceamento para cada video.",
    )
    parser.add_argument(
        "--link-type",
        choices=["hardlink", "symlink"],
        default="hardlink",
        help="hardlink preserva o nome no CVAT sem duplicar dados; symlink pode ser resolvido para o nome original.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stems = args.only_stem if args.only_stem else VIDEOS
    balanced_root = Path(args.balanced_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    if args.force:
        clean_root(balanced_root)
    else:
        balanced_root.mkdir(parents=True, exist_ok=True)

    if args.split_per_video:
        chunks = []
        for stem, intervals in build_intervals_by_video(
            stems,
            Path(args.selection_root),
            Path(args.frames_root),
            args.window,
        ).items():
            for local_index, chunk in enumerate(split_chunks(intervals, args.min_task_size), start=1):
                chunks.append((stem, local_index, chunk))
    else:
        intervals = build_intervals(
            stems,
            Path(args.selection_root),
            Path(args.frames_root),
            args.window,
        )
        chunks = split_chunks(intervals, args.min_task_size)
    rows = create_chunk_links(chunks, args)

    summary_path = results_root / f"{args.task_prefix}_summary_{int(time.time())}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_prefix": args.task_prefix,
                "min_task_size": args.min_task_size,
                "window": args.window,
                "split_per_video": args.split_per_video,
                "link_type": args.link_type,
                "chunks": rows,
                "total_frames": sum(row["frames"] for row in rows),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
