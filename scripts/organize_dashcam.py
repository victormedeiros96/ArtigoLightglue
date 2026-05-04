import os
import shutil
from collections import defaultdict
from tqdm import tqdm

def organize_bdd100k(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Agrupar por prefixo
    groups = defaultdict(list)
    files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    
    print(f"Agrupando {len(files)} imagens...")
    for f in files:
        parts = f.split('-')
        if len(parts) == 2:
            seq_id = parts[0]
            groups[seq_id].append(f)
            
    print(f"Encontradas {len(groups)} sequências.")
    
    # Criar estrutura de pastas com symlinks
    for seq_id, frames in tqdm(groups.items(), desc="Organizando"):
        seq_path = os.path.join(dest_dir, seq_id)
        if not os.path.exists(seq_path):
            os.makedirs(seq_path)
        
        # Ordenar frames (o BDD100K usa hashes mas eles costumam ter ordem alfanumérica consistente)
        frames.sort()
        
        for i, f in enumerate(frames):
            src_file = os.path.abspath(os.path.join(src_dir, f))
            # Nomeamos como frame_0000.jpg para facilitar o tracking
            dest_file = os.path.join(seq_path, f"frame_{i:04d}.jpg")
            if not os.path.exists(dest_file):
                os.symlink(src_file, dest_file)

if __name__ == "__main__":
    src = "/mnt/hd2/ArtigoLightglue/Datasets_externos/dashcam_extracted/test"
    dest = "/mnt/hd2/ArtigoLightglue/Datasets_externos/dashcam_organized"
    organize_bdd100k(src, dest)
