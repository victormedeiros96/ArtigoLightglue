import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/servidor/ArtigoLightglue")
BALANCED_ROOT = Path("/mnt/hd2/tasks_stag_go_pro_dez/cam1_plate_balanced_frames")
RESULTS_ROOT = ROOT / "results/plate_balanced_chunks"

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
            f"Command failed ({proc.returncode}): {' '.join(cmd[:16])} ...\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return "\n".join(part for part in [proc.stdout, proc.stderr] if part).strip()


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
        name = task.get("name", "")
        if name.startswith(args.task_prefix):
            mapping[name] = int(task["id"])
    return mapping


def delete_task(args, task_id):
    run_cmd(cvat_base_cmd(args) + ["task", "delete", str(task_id)], env={"PASS": args.cvat_pass})


def list_chunk_dirs(args):
    root = Path(args.balanced_root)
    return sorted(path for path in root.glob(f"{args.task_prefix}_*") if path.is_dir())


def list_images(chunk_dir):
    return sorted(chunk_dir.glob("*.jpg"))


def parse_task_id(output):
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


def create_task(args, chunk_dir):
    files = list_images(chunk_dir)
    if not files:
        raise RuntimeError(f"Nenhum JPG encontrado em {chunk_dir}")
    task_name = chunk_dir.name
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
        *[str(path) for path in files],
    ]
    if args.dry_run:
        print(f"[DRY-RUN] {' '.join(cmd[:16])} ... {len(files)} files")
        return None
    output = run_cmd(cmd, env={"PASS": args.cvat_pass})
    return parse_task_id(output)


def main():
    parser = argparse.ArgumentParser(description="Cria tasks CVAT para chunks balanceados.")
    parser.add_argument("--balanced-root", default=str(BALANCED_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--task-prefix", default="cam1_plate_balanced")
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

    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    existing = get_existing_tasks(args)
    rows = []
    for chunk_dir in list_chunk_dirs(args):
        task_name = chunk_dir.name
        n_frames = len(list_images(chunk_dir))
        if task_name in existing:
            if not args.force_create:
                task_id = existing[task_name]
                print(f"[SKIP] {task_name} ja existe: task={task_id}")
                rows.append({"task_name": task_name, "task_id": task_id, "frames": n_frames})
                continue
            if not args.dry_run:
                print(f"[DELETE] {task_name} task={existing[task_name]}")
                delete_task(args, existing[task_name])

        print(f"[CREATE] {task_name} frames={n_frames}")
        task_id = create_task(args, chunk_dir)
        print(f"[DONE] {task_name} task={task_id}")
        rows.append({"task_name": task_name, "task_id": task_id, "frames": n_frames})

    out_path = Path(args.results_root) / f"{args.task_prefix}_task_map_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[SUMMARY] {out_path}")


if __name__ == "__main__":
    main()
