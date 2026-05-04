import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import shutil

def parse_cvat_video_xml(xml_path, out_gt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    gt_lines = []
    
    for track in root.findall('track'):
        track_id = int(track.get('id'))
        
        for box in track.findall('box'):
            # Only consider frames where the object is NOT outside
            outside = int(box.get('outside', '0'))
            if outside == 1:
                continue
                
            frame = int(box.get('frame')) + 1 # MOT uses 1-based frames
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            w = xbr - xtl
            h = ybr - ytl
            
            # format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            line = f"{frame},{track_id},{xtl:.2f},{ytl:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
            gt_lines.append((frame, line))
            
    # Sort by frame
    gt_lines.sort(key=lambda x: x[0])
    
    with open(out_gt_path, 'w') as f:
        for _, line in gt_lines:
            f.write(line)

def main():
    BASE_DIR = Path("/mnt/hd2/ArtigoLightglue/Datasets_externos/cvat_plates")
    BENCHMARK_DIR = Path("/mnt/hd2/ArtigoLightglue/Datasets_externos/gopro_benchmark")
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_files = sorted(BASE_DIR.glob('*.zip'))
    
    for zip_path in tqdm(zip_files, desc="Extracting and Converting CVAT datasets"):
        seq_name = zip_path.stem
        seq_dir = BENCHMARK_DIR / seq_name
        
        # Unzip if not already unzipped
        if not seq_dir.exists():
            print(f"\nUnzipping {seq_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(seq_dir)
                
        # Move images folder to img1 (MOT format)
        images_src = seq_dir / 'images'
        img1_dst = seq_dir / 'img1'
        if images_src.exists() and not img1_dst.exists():
            images_src.rename(img1_dst)
            
        # Parse XML to gt.txt
        gt_dir = seq_dir / 'gt'
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_txt = gt_dir / 'gt.txt'
        
        xml_path = seq_dir / 'annotations.xml'
        if xml_path.exists():
            parse_cvat_video_xml(xml_path, gt_txt)
        else:
            print(f"Warning: annotations.xml not found in {seq_name}")
            
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
