def get_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

class NaiveTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 1
        self.active_tracks = {} # id -> last_box
        self.threshold = iou_threshold

    def update(self, detections):
        new_ids = []
        current_active = {}
        for det_box in detections:
            best_iou = 0
            matched_id = -1
            for obj_id, last_box in self.active_tracks.items():
                iou = get_iou(det_box, last_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_id = obj_id
            if best_iou >= self.threshold:
                new_ids.append(matched_id)
                current_active[matched_id] = det_box
            else:
                new_ids.append(self.next_id)
                current_active[self.next_id] = det_box
                self.next_id += 1
        self.active_tracks = current_active
        return new_ids
