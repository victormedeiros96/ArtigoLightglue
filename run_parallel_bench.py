import subprocess
import os
import time

# USE THE 1080P PROXY FOR SPEED AND FAIRNESS (MATCHES LIGHTGLUE RUN)
video = "/home/servidor/ArtigoLightglue/segment_1080p_120s.mp4"
model = "/home/servidor/ArtigoLightglue/detector_placa.pt"
trackers = ["botsort.yaml", "bytetrack.yaml"]
framerates = [30, 5, 1]
max_frames = 3600 # 120 seconds

cmds = []
gpu_id = 0
for fps in framerates:
    stride = 30 // fps
    for t in trackers:
        tracker_file = t
        if fps < 30:
            tracker_file = t.replace(".yaml", "_lowfps.yaml")
        exp_name = f"{t.split('.')[0]}_{fps}fps"
        cmd = f"python3 /home/servidor/ArtigoLightglue/run_single_track.py --video {video} --model {model} --tracker {tracker_file} --stride {stride} --name {exp_name} --device {gpu_id} --max_frames {max_frames}"
        cmds.append(cmd)
        gpu_id = (gpu_id + 1) % 4

print(f"Starting experiments for {max_frames} frames on 1080p proxy...")
processes = [subprocess.Popen(c.split()) for c in cmds]
for p in processes:
    p.wait()
print("All experiments finished.")
