#DETECTION


import numpy as np
import pandas as pd
!pip install -q ultralytics
import torch
print("GPU Available:", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using:", device)

from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/kaggle/input/datasets/rahilvats/football-dataset/data.yaml",
    epochs=50,
    imgsz=1280,
    batch=8,
    device=[0,1],
    workers=2
)


model.val()


preds = model.predict(
    source="/kaggle/input/datasets/rahilvats/football-dataset/valid/images",
    save=True
)

print("Training complete. Check /kaggle/working/runs/detect/train")



#TRACKING

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque


MODEL_PATH = "/kaggle/working/runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)


BALL    = 0
GK      = 1
PLAYER  = 2
REFEREE = 3


tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    embedder="mobilenet",
    half=True,
    bgr=True,
)


def match_yolo_box(track_tlbr, yolo_boxes, yolo_cls, cls_filter):
    tx1, ty1, tx2, ty2 = track_tlbr
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0
    best_box  = None
    best_dist = 80.0
    for box, cls in zip(yolo_boxes, yolo_cls):
        if cls != cls_filter:
            continue
        bx1, by1, bx2, by2 = box
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        d = np.hypot(tcx - bcx, tcy - bcy)
        if d < best_dist:
            best_dist = d
            best_box  = (int(bx1), int(by1), int(bx2), int(by2))
    return best_box



PITCH_W, PITCH_H = 105.0, 68.0
PA = 16.5           # penalty area depth
PB_HALF = 27.16 / 2 # penalty box half-width = 13.58 m → full width 27.16 m
GA_HALF = 7.32 / 2  # goal area half = 5.5 m → 11.0 m wide
GA_D    = 5.5       # goal area depth

PITCH_LANDMARKS = np.array([
    # Corners
    [0,         0       ],
    [PITCH_W,   0       ],
    [0,         PITCH_H ],
    [PITCH_W,   PITCH_H ],
    # Halfway line × touchlines
    [PITCH_W/2, 0       ],
    [PITCH_W/2, PITCH_H ],
    # Left penalty box corners
    [0,         PITCH_H/2 - PB_HALF],
    [PA,        PITCH_H/2 - PB_HALF],
    [PA,        PITCH_H/2 + PB_HALF],
    [0,         PITCH_H/2 + PB_HALF],
    # Right penalty box corners
    [PITCH_W,      PITCH_H/2 - PB_HALF],
    [PITCH_W - PA, PITCH_H/2 - PB_HALF],
    [PITCH_W - PA, PITCH_H/2 + PB_HALF],
    [PITCH_W,      PITCH_H/2 + PB_HALF],
    # Left goal-area corners
    [0,      PITCH_H/2 - GA_HALF],
    [GA_D,   PITCH_H/2 - GA_HALF],
    [GA_D,   PITCH_H/2 + GA_HALF],
    [0,      PITCH_H/2 + GA_HALF],
    # Right goal-area corners
    [PITCH_W,         PITCH_H/2 - GA_HALF],
    [PITCH_W - GA_D,  PITCH_H/2 - GA_HALF],
    [PITCH_W - GA_D,  PITCH_H/2 + GA_HALF],
    [PITCH_W,         PITCH_H/2 + GA_HALF],
], dtype=np.float32)


def get_pitch_mask(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35,  30,  40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    return mask


def detect_pitch_lines(frame, mask):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray   = cv2.bitwise_and(gray, gray, mask=mask)
    edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
    edges  = cv2.bitwise_and(edges, edges, mask=eroded)
    lines  = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi/180,
        threshold=60,
        minLineLength=60,
        maxLineGap=20,
    )
    if lines is None:
        return []
    return [tuple(l[0]) for l in lines]


def line_angle_deg(x1, y1, x2, y2):
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))


def segment_lines(lines):
    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines:
        a = line_angle_deg(x1, y1, x2, y2)
        if a < 30:
            horizontals.append((x1, y1, x2, y2))
        elif a > 60:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals


def line_to_params(x1, y1, x2, y2):
    a = float(y2 - y1)
    b = float(x1 - x2)
    c = float(x2 * y1 - x1 * y2)
    norm = np.hypot(a, b)
    if norm < 1e-6:
        return None
    return a / norm, b / norm, c / norm


def intersection_point(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return float(x), float(y)


def compute_intersections(horizontals, verticals, frame_shape):
    H, W = frame_shape[:2]
    h_params = [line_to_params(*l) for l in horizontals]
    v_params = [line_to_params(*l) for l in verticals]
    pts = []
    for hp in h_params:
        if hp is None:
            continue
        for vp in v_params:
            if vp is None:
                continue
            pt = intersection_point(hp, vp)
            if pt is None:
                continue
            x, y = pt
            if 5 < x < W - 5 and 5 < y < H - 5:
                pts.append([x, y])
    return np.array(pts, dtype=np.float32) if pts else None


def match_intersections_to_template(detected_pts, frame_shape):
    if detected_pts is None or len(detected_pts) < 4:
        return None

    H_frame, W_frame = frame_shape[:2]
    norm_detected = detected_pts / np.array([W_frame, H_frame], dtype=np.float32)
    norm_template = PITCH_LANDMARKS / np.array([PITCH_W, PITCH_H], dtype=np.float32)


    src_pts = []
    dst_pts = []
    used_template = set()
    for i, nd in enumerate(norm_detected):
        dists = np.linalg.norm(norm_template - nd, axis=1)
        for j in np.argsort(dists):
            if j in used_template:
                continue
            if dists[j] < 0.15:   
                src_pts.append(detected_pts[i])
                dst_pts.append(PITCH_LANDMARKS[j])
                used_template.add(j)
            break

    if len(src_pts) < 4:
        return None

    src_np = np.array(src_pts, dtype=np.float32)
    dst_np = np.array(dst_pts, dtype=np.float32)

    H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 3.0)
    if H is None:
        return None

    inlier_count = int(mask.sum()) if mask is not None else 0
    if inlier_count < 4:
        return None

    return H


class AutoHomography:
    RECOMPUTE_INTERVAL = 15   
    ALPHA              = 0.25  
    def __init__(self):
        self._H       = None  
        self._counter = 0
        self.valid    = False

    def update(self, frame):
        self._counter += 1
        if self._counter % self.RECOMPUTE_INTERVAL != 1:
            return self.valid    
        mask  = get_pitch_mask(frame)
        lines = detect_pitch_lines(frame, mask)
        if not lines:
            return self.valid

        h_lines, v_lines = segment_lines(lines)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return self.valid

        int_pts = compute_intersections(h_lines, v_lines, frame.shape)
        H_new   = match_intersections_to_template(int_pts, frame.shape)

        if H_new is None:
            return self.valid

        if self._H is None:
            self._H = H_new
        else:
            self._H = (1 - self.ALPHA) * self._H + self.ALPHA * H_new

        self.valid = True
        return True

    def pixel_to_metres(self, px, py):
        if not self.valid or self._H is None:
            return None
        pt     = np.array([[[float(px), float(py)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        return float(result[0][0][0]), float(result[0][0][1])


AVG_PLAYER_HEIGHT_M = 1.75
_scale_buf = deque(maxlen=30)

def estimate_scale_fallback(yolo_boxes, yolo_cls):
    heights = [
        (b[3] - b[1])
        for b, c in zip(yolo_boxes, yolo_cls)
        if c == PLAYER and (b[3] - b[1]) > 20
    ]
    if not heights:
        return float(np.mean(_scale_buf)) if _scale_buf else None
    s = AVG_PLAYER_HEIGHT_M / float(np.median(heights))
    _scale_buf.append(s)
    return float(np.mean(_scale_buf))


class SpeedEstimator:
    
    def __init__(self, smoothing=5, max_speed_kmh=45.0):
        self.last_pos      = {}   # tid → (mx, my, t)
        self.speed_buf     = defaultdict(lambda: deque(maxlen=smoothing))
        self.max_speed     = max_speed_kmh

    def update(self, tid, px, py, t, auto_hom: AutoHomography,
               fallback_mpp=None):
        pos = auto_hom.pixel_to_metres(px, py)
        if pos is not None:
            mx, my = pos
        elif fallback_mpp is not None:
            mx, my = px * fallback_mpp, py * fallback_mpp
        else:
            return 0.0

        if tid not in self.last_pos:
            self.last_pos[tid] = (mx, my, t)
            return 0.0

        xp, yp, tp = self.last_pos[tid]
        dt = t - tp
        if dt < 1e-4:
            return float(np.mean(self.speed_buf[tid])) if self.speed_buf[tid] else 0.0

        dist_m     = np.hypot(mx - xp, my - yp)
        speed_kmph = min((dist_m / dt) * 3.6, self.max_speed)

        self.last_pos[tid] = (mx, my, t)
        self.speed_buf[tid].append(speed_kmph)
        return float(np.mean(self.speed_buf[tid]))


player_speed_est = SpeedEstimator(smoothing=5, max_speed_kmh=45.0)
ball_speed_est   = SpeedEstimator(smoothing=3, max_speed_kmh=250.0)


CLASS_COLORS = {
    BALL:   (0, 165, 255),
    GK:     (0, 255, 255),
    PLAYER: (0, 255,   0),
}

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    auto_hom = AutoHomography()
    frame_id  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_id / fps

        #  Update homography (automatic, every 15 frames) 
        hom_valid = auto_hom.update(frame)

        # YOLO 
        results = model(frame, conf=0.3, imgsz=1280, verbose=False)[0]
        xyxy  = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy().astype(int)

        mask = clss != REFEREE
        xyxy, confs, clss = xyxy[mask], confs[mask], clss[mask]

        fallback_mpp = None if hom_valid else estimate_scale_fallback(xyxy, clss)

        # DeepSORT 
        raw_dets = [
            [[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
             float(c), int(cl)]
            for b, c, cl in zip(xyxy, confs, clss)
        ]
        tracks = tracker.update_tracks(raw_dets, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            cls = int(t.det_class) if t.det_class is not None else PLAYER
            if cls == REFEREE:
                continue

            tid   = t.track_id
            color = CLASS_COLORS.get(cls, (180, 180, 180))

            yolo_box = match_yolo_box(t.to_tlbr(), xyxy, clss, cls)
            if yolo_box is None:
                continue   

            x1, y1, x2, y2 = yolo_box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2   

            ground_x = cx
            ground_y = y2 if cls in (PLAYER, GK) else cy

            # Speed
            if cls == BALL:
                speed = ball_speed_est.update(
                    tid, ground_x, ground_y, ts, auto_hom, fallback_mpp)
                label = f"Ball  {speed:.0f} km/h"
            else:
                speed = player_speed_est.update(
                    tid, ground_x, ground_y, ts, auto_hom, fallback_mpp)
                role  = "GK" if cls == GK else f"P{tid}"
                label = f"{role}  {speed:.0f} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            font_s = 0.40
            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_s, 1)
            lx = max(x1, 0)
            ly = max(y1 - th - bl - 6, 0)
            cv2.rectangle(frame,
                          (lx,      ly),
                          (lx + tw + 8, ly + th + bl + 4),
                          (15, 15, 15), -1)
            cv2.rectangle(frame,
                          (lx,      ly),
                          (lx + 3,  ly + th + bl + 4),
                          color, -1)
            cv2.putText(frame, label, (lx + 6, ly + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_s,
                        color, 1, cv2.LINE_AA)

            if cls == BALL:
                cv2.circle(frame, (cx, cy), 5, color, -1)

        cal_text  = "CAL: homography" if hom_valid else "CAL: player-height"
        cal_color = (0, 255, 150) if hom_valid else (0, 165, 255)
        cv2.putText(frame, cal_text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cal_color, 1)

        cv2.putText(frame, f"{ts:.1f}s",
                    (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (160, 160, 160), 1)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[DONE] {frame_id} frames → {output_path}")


if __name__ == "__main__":
    input_video  = (
        "/kaggle/input/datasets/rahilvats/football-videos/"
        "stock-footage-edited-montage-of-a-nostalgic-soccer-championship-match-with-score-teams-attack-score-goal.mp4"
    )
    output_video = "/kaggle/working/output_v4.mp4"
    process_video(input_video, output_video)
