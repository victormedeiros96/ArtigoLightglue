import os
import subprocess
from tqdm import tqdm

VIDEOS = [
    "GX010084.MP4", "GX010083.MP4", "GX010069.MP4", 
    "GX010076.MP4", "GX010086.MP4", "GX010067.MP4", 
    "GX010080.MP4", "GX010081.MP4", "GX010085.MP4"
]

SRC_DIR = "/mnt/hd2/tasks_stag_go_pro_dez/cam1"
DST_DIR = "/home/servidor/ArtigoLightglue/proxies"
os.makedirs(DST_DIR, exist_ok=True)

for v_name in tqdm(VIDEOS, desc="Convertendo Vídeos"):
    v_src = os.path.join(SRC_DIR, v_name)
    v_dst = os.path.join(DST_DIR, v_name)
    
    if os.path.exists(v_dst) and os.path.getsize(v_dst) > 1000000:
        print(f"PULANDO: {v_name} já existe.")
        continue
        
    print(f"\nProcessando {v_name}...")
    # -t 120 (primeiros 2 min), -vf scale=1920:1080, -preset ultrafast (velocidade máxima)
    cmd = [
        "ffmpeg", "-y", "-i", v_src, 
        "-t", "120", "-vf", "scale=1920:1080", 
        "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast", 
        v_dst
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(v_dst):
        size = os.path.getsize(v_dst) / (1024*1024)
        print(f"CONCLUÍDO: {v_name} ({size:.2f} MB)")
    else:
        print(f"ERRO ao converter {v_name}")

print("\n--- Todos os proxies foram gerados! ---")
