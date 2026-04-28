import torch
import numpy as np
import cv2
from collections import deque
from scipy.optimize import linear_sum_assignment
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch

class Track:
    def __init__(self, bbox, features, cls, tid, frame_idx):
        self.bbox = bbox
        self.features = features
        self.cls = cls
        self.id = tid
        self.last_frame = frame_idx
        self.age = 0 # Frames desde a última detecção
        self.center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.velocity = np.array([0.0, 0.0])
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.img_center = np.array([960, 540])
        self.radial_dist = np.linalg.norm(self.center - self.img_center)

class LightGlueTracker:
    def __init__(self, device='cuda', accept_th=1.2, motion_weight=0.5, max_age=5, min_matches_short=15, min_matches_global=30, verbose=False):
        self.device = device
        # Rigor de Inventário: Apenas 5 frames de tolerância em 30fps (0.16s)
        self.max_age_frames = max_age 
        self.min_matches_short = min_matches_short
        self.min_matches_global = min_matches_global
        self.accept_th = accept_th
        self.motion_weight = motion_weight
        self.verbose = verbose
        
        self.next_id = 1
        self.tracks = deque()
        self.gallery = {}
        self.match_stats = []
        self.img_center = np.array([960, 540])
        
        # SuperPoint calibrado para 1024 pontos para cobrir o contexto ampliado
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
        self.matcher = LightGlue(features='superpoint', filter_threshold=0.2).eval().to(device)

    def get_ultra_context_crop(self, img, box, factor=5.0):
        """Extrai crop contextual sem filtros de processamento (natural)"""
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nw, nh = int(w * factor), int(h * factor)
        H, W = img.shape[:2]
        nx1, nx2 = max(0, cx - nw // 2), min(W, cx + nw // 2)
        ny1, ny2 = max(0, cy - nh // 2), min(H, cy + nh // 2)
        crop = img[ny1:ny2, nx1:nx2]
        if crop.size == 0: return np.zeros((320, 320, 3), dtype=np.uint8)
        # Removido CLAHE para manter a fidelidade original da GoPro
        return cv2.resize(crop, (320, 320))

    def extract_features_single(self, frame, bboxes):
        feats = []
        with torch.no_grad():
            for box in bboxes:
                img = self.get_ultra_context_crop(frame, box)
                t = numpy_image_to_torch(img).to(self.device).float()
                f = self.extractor.extract(t) 
                feats.append(f)
        return feats

    def pad_and_batch(self, feat_list):
        if not feat_list: return {}
        max_k = max(f['keypoints'].shape[1] for f in feat_list)
        batch = {}
        for k in feat_list[0].keys():
            if k == 'image_size':
                batch[k] = torch.cat([f[k] for f in feat_list], dim=0)
                continue
            padded = []
            for f in feat_list:
                tensor = f[k]
                p_size = max_k - tensor.shape[1]
                if p_size > 0:
                    p_shape = list(tensor.shape); p_shape[1] = p_size
                    padded.append(torch.cat([tensor, torch.zeros(p_shape, device=self.device, dtype=tensor.dtype)], dim=1))
                else: padded.append(tensor)
            batch[k] = torch.cat(padded, dim=0)
        return batch

    def update(self, bboxes, frame, classes, current_frame, stride=1):
        # Escala da tolerância de apagão baseada no FPS alvo (stride)
        allowed_gap = max(1, 5 // stride) # Ex: 30fps permite 5 frames, 5fps permite 1 frame.
        self.age_tracks(allowed_gap)
        
        if len(bboxes) == 0: return []
        H, W = frame.shape[:2]
        
        # CLIPPING GUARD RÍGIDO (Detectou corte na borda = descarta)
        valid_indices = []
        for i, b in enumerate(bboxes):
            x1, y1, x2, y2 = b
            if x1 <= 1 or y1 <= 1 or x2 >= (W - 1) or y2 >= (H - 1):
                continue
            valid_indices.append(i)
        if not valid_indices: return []
        
        bboxes = [bboxes[i] for i in valid_indices]
        classes = [classes[i] for i in valid_indices]

        det_features = self.extract_features_single(frame, bboxes)
        det_centers = [np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]) for b in bboxes]
        det_radial_dists = [np.linalg.norm(c - self.img_center) for c in det_centers]
        
        matched_detect_indices, active_results = set(), []
        
        # Gate espacial: cap em 500px independente do stride
        # (evita que 200*stride=1200px a 5fps cruce metade da imagem)
        max_spatial_dist = min(200 * stride, 500)
        half_W = W / 2
        center_tolerance = 120  # px de tolerância ao redor do centro
        if self.tracks and det_features:
            valid_pairs = []
            for d_idx in range(len(det_features)):
                for t_idx, trk in enumerate(self.tracks):
                    if classes[d_idx] != trk.cls: continue
                    
                    # Lei Radial: objeto estático só se afasta do centro
                    if det_radial_dists[d_idx] < trk.radial_dist - 30: continue
                    
                    # Gate Euclidiano: rejeita se longe demais
                    dist = np.linalg.norm(det_centers[d_idx] - trk.center)
                    if dist > max_spatial_dist: continue
                    
                    # Guarda Esquerda/Direita: placa não teleporta entre lados
                    # Só bloqueia se AMBOS estiverem longe do centro (fora da zona de tolerância)
                    det_x = det_centers[d_idx][0]
                    trk_x = trk.center[0]
                    det_near_center = abs(det_x - half_W) < center_tolerance
                    trk_near_center = abs(trk_x - half_W) < center_tolerance
                    det_left = det_x < half_W
                    trk_left = trk_x < half_W
                    if (det_left != trk_left) and not det_near_center and not trk_near_center:
                        continue  # Lados opostos, ambos longe do centro = impossível
                    
                    valid_pairs.append((d_idx, t_idx))
            
            if valid_pairs:
                b0 = self.pad_and_batch([self.tracks[p[1]].features for p in valid_pairs])
                b1 = self.pad_and_batch([det_features[p[0]] for p in valid_pairs])
                with torch.no_grad(): matches = self.matcher({'image0': b0, 'image1': b1})
                
                cost_matrix = np.ones((len(det_features), len(self.tracks))) * 5.0
                for i, (d_idx, t_idx) in enumerate(valid_pairs):
                    num_m = (matches['matches0'][i] > -1).sum().item()
                    if num_m >= 15:
                        trk = self.tracks[t_idx]
                        visual_conf = min(1.0, num_m / 150.0)
                        dt = current_frame - trk.last_frame
                        expected_center = trk.center + (trk.velocity * dt)
                        pred_error = np.linalg.norm(det_centers[d_idx] - expected_center) / 1920.0
                        radial_penalty = 0.5 if det_radial_dists[d_idx] < trk.radial_dist else 0.0
                        # Custo espacial normalizado: penaliza detecções fisicamente distantes
                        dist = np.linalg.norm(det_centers[d_idx] - trk.center)
                        spatial_cost = dist / max_spatial_dist  # 0=perto, 1=limite
                        cost_matrix[d_idx, t_idx] = (1.0 - visual_conf) + (pred_error * self.motion_weight) + radial_penalty + (spatial_cost * 0.4)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < self.accept_th:
                        trk = self.tracks[c]
                        dt = current_frame - trk.last_frame
                        if dt > 0:
                            new_vel = (det_centers[r] - trk.center) / dt
                            trk.velocity = 0.5 * trk.velocity + 0.5 * new_vel
                        trk.bbox, trk.center, trk.features, trk.last_frame, trk.age = bboxes[r], det_centers[r], det_features[r], current_frame, 0
                        trk.radial_dist = det_radial_dists[r]
                        matched_detect_indices.add(r); active_results.append({"id": trk.id, "bbox": bboxes[r], "cls": trk.cls})

        for i, det_feat in enumerate(det_features):
            if i in matched_detect_indices: continue
            # Cadastro de Ativo: Novo objeto ganha ID imediatamente.
            new_id = self.next_id; self.next_id += 1
            self.gallery[new_id] = {"features": det_feat, "cls": classes[i], "frame": current_frame, "center": det_centers[i], "velocity": np.array([0.0, 0.0]), "radial_dist": det_radial_dists[i]}
            self.tracks.append(Track(bboxes[i], det_feat, classes[i], new_id, current_frame))
            active_results.append({"id": new_id, "bbox": bboxes[i], "cls": classes[i]})
        return active_results

    def age_tracks(self, allowed_gap):
        """Mata rastros que sumiram por mais tempo do que o allowed_gap"""
        # Aumentar a idade de todos
        for t in self.tracks: t.age += 1
        
        # Remover os mais velhos da frente da fila
        while self.tracks and self.tracks[0].age > allowed_gap:
            trk = self.tracks.popleft()
            # Guardamos na galeria mas não usamos para ReID automático agressivo por enquanto
            self.gallery[trk.id].update({"center": trk.center, "frame": trk.last_frame, "velocity": trk.velocity, "radial_dist": trk.radial_dist})
