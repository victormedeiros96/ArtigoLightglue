import argparse
import json
import os
import subprocess
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
SELECTED_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_plate_selected_frames")
RESULTS_ROOT = ROOT / "results/plate_frame_selection"

CVAT_HOST = "http://192.168.18.140"
CVAT_PORT = "8080"
CVAT_ORG = "RDT3"
CVAT_PROJECT_ID = 79
CVAT_USER = "superrdt"
CVAT_PASS = "superpwdrdt"


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
    return proc.stdout.strip()


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


def get_existing_tasks(args):
    output = run_cmd(cvat_base_cmd(args) + ["task", "ls", "--json"], env={"PASS": args.cvat_pass})
    tasks = json.loads(output)
    mapping = {}
    for task in tasks:
        if task.get("project_id") != args.project_id:
            continue
        mapping[task.get("name", "")] = int(task["id"])
    return mapping


def count_frames(frame_dir, stem):
    return len(list(frame_dir.glob(f"{stem}_frame_*.jpg")))


def list_frame_files(frame_dir, stem):
    return sorted(frame_dir.glob(f"{stem}_frame_*.jpg"))


def delete_task(args, task_id):
    run_cmd(cvat_base_cmd(args) + ["task", "delete", str(task_id)], env={"PASS": args.cvat_pass})


def create_task(args, stem, frame_dir):
    task_name = f"{stem}{args.task_suffix}"
    frame_files = list_frame_files(frame_dir, stem)
    if not frame_files:
        raise RuntimeError(f"Nenhum JPG encontrado para {stem} em {frame_dir}")

    cmd = cvat_base_cmd(args) + [
        "task",
        "create",
        "--project_id",
        str(args.project_id),
        "--image_quality",
        str(args.image_quality),
        "--sorting-method",
        "natural",
        task_name,
        "local",
        *[str(path) for path in frame_files],
    ]
    if args.dry_run:
        preview = " ".join(cmd[:16])
        print(f"[DRY-RUN] {preview} ... {len(frame_files)} files")
        return None
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PASS": args.cvat_pass},
    )
    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd[:16])} ... {len(frame_files)} files\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    # cvat-cli usually prints the task id as the last non-empty line.
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.isdigit():
            return int(line)
        marker = "Created task ID:"
        if marker in line:
            tail = line.split(marker, 1)[1].strip().split()
            if tail and tail[0].isdigit():
                return int(tail[0])
    raise RuntimeError(f"Nao consegui extrair task id da saida:\n{output}")


def main():
    parser = argparse.ArgumentParser(
        description="Cria tasks CVAT a partir das pastas de symlinks selecionadas."
    )
    parser.add_argument("--selected-root", default=str(SELECTED_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--only-stem", nargs="*", default=None)
    parser.add_argument("--task-suffix", default="_plate_frames")
    parser.add_argument("--project-id", type=int, default=CVAT_PROJECT_ID)
    parser.add_argument("--image-quality", type=int, default=100)
    parser.add_argument("--force-create", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cvat-host", default=CVAT_HOST)
    parser.add_argument("--cvat-port", default=CVAT_PORT)
    parser.add_argument("--cvat-org", default=CVAT_ORG)
    parser.add_argument("--cvat-user", default=CVAT_USER)
    parser.add_argument("--cvat-pass", default=CVAT_PASS)
    args = parser.parse_args()

    selected_root = Path(args.selected_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    stems = args.only_stem if args.only_stem else VIDEOS
    existing = get_existing_tasks(args)
    rows = []

    for stem in stems:
        frame_dir = selected_root / stem
        n_frames = count_frames(frame_dir, stem) if frame_dir.exists() else 0
        if n_frames == 0:
            print(f"[SKIP] {stem}: pasta vazia ou ausente: {frame_dir}")
            continue

        task_name = f"{stem}{args.task_suffix}"
        if not args.force_create and task_name in existing:
            task_id = existing[task_name]
            print(f"[SKIP] {task_name} ja existe: task={task_id}")
        else:
            if args.force_create and task_name in existing and not args.dry_run:
                old_task_id = existing[task_name]
                print(f"[DELETE] {task_name} task={old_task_id}")
                delete_task(args, old_task_id)
            print(f"[CREATE] {task_name} frames={n_frames}")
            task_id = create_task(args, stem, frame_dir)
            print(f"[DONE] {task_name} task={task_id}")

        rows.append(
            {
                "video": stem,
                "task_name": task_name,
                "task_id": task_id,
                "frames": n_frames,
                "frame_dir": str(frame_dir),
            }
        )

    out_path = results_root / f"cvat_plate_task_map_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[SUMMARY] {out_path}")


if __name__ == "__main__":
    main()
