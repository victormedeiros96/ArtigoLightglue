import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_uavdt(root_dir, out_dir):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        split_dir = root_dir / split
        img_dir = split_dir / 'img'
        ann_dir = split_dir / 'ann'
        
        if not img_dir.exists():
            continue
            
        print(f"Processing split: {split}")
        
        # Get all annotation files
        ann_files = sorted(list(ann_dir.glob('*.json')))
        
        sequences = {}
        
        for ann_path in tqdm(ann_files, desc=f"Reading annotations {split}"):
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Extract sequence name from tags or filename
            seq_name = None
            for tag in data.get('tags', []):
                if tag['name'] == 'sequence':
                    seq_name = tag['value']
                    break
            
            if not seq_name:
                # Fallback to filename prefix: M0101_img000001.jpg.json -> M0101
                seq_name = ann_path.name.split('_')[0]
            
            if seq_name not in sequences:
                sequences[seq_name] = []
            
            # Extract frame number from filename
            # M0101_img000001.jpg.json -> 1
            try:
                frame_id_str = ann_path.name.split('_img')[-1].split('.')[0]
                frame_id = int(frame_id_str)
            except:
                print(f"Warning: Could not parse frame ID from {ann_path.name}")
                continue
                
            frame_data = {
                'frame_id': frame_id,
                'img_name': ann_path.name.replace('.json', ''),
                'objects': []
            }
            
            for obj in data.get('objects', []):
                if obj['geometryType'] != 'rectangle':
                    continue
                
                target_id = None
                for tag in obj.get('tags', []):
                    if tag['name'] == 'target id':
                        target_id = tag['value']
                        break
                
                if target_id is None:
                    continue
                
                # points.exterior is [[x1, y1], [x2, y2]]
                points = obj['points']['exterior']
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                w = x2 - x1
                h = y2 - y1
                
                frame_data['objects'].append({
                    'id': target_id,
                    'bbox': [x1, y1, w, h],
                    'class': obj['classTitle']
                })
            
            sequences[seq_name].append(frame_data)
            
        # Write organized structure
        for seq_name, frames in sequences.items():
            seq_out_dir = out_dir / seq_name
            seq_img_dir = seq_out_dir / 'img1'
            seq_gt_dir = seq_out_dir / 'gt'
            
            seq_img_dir.mkdir(parents=True, exist_ok=True)
            seq_gt_dir.mkdir(parents=True, exist_ok=True)
            
            # Sort frames by ID
            frames.sort(key=lambda x: x['frame_id'])
            
            gt_path = seq_gt_dir / 'gt.txt'
            with open(gt_path, 'w') as f_gt:
                for frame in frames:
                    # Symlink image
                    src_img = img_dir / frame['img_name']
                    dst_img = seq_img_dir / f"{frame['frame_id']:06d}.jpg"
                    
                    if not dst_img.exists():
                        try:
                            os.symlink(src_img, dst_img)
                        except FileExistsError:
                            pass
                    
                    # Write GT line: <frame>, <id>, <x>, <y>, <w>, <h>, 1, -1, -1, -1
                    for obj in frame['objects']:
                        bbox = obj['bbox']
                        f_gt.write(f"{frame['frame_id']},{obj['id']},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,-1,-1,-1\n")

    print("Organization complete!")

if __name__ == "__main__":
    UAVDT_ROOT = "/mnt/hd2/ArtigoLightglue/Datasets_externos/uavdt_full"
    UAVDT_ORG = "/mnt/hd2/ArtigoLightglue/Datasets_externos/uavdt_organized"
    organize_uavdt(UAVDT_ROOT, UAVDT_ORG)
