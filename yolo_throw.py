"""
Strict THROW Detection Module
-----------------------------
- Detects if small fast objects are thrown (off-hand, airborne).
- Uses YOLO detections + centroid tracker + motion checks.
- Filters out cases when object is still in hand/overlaps a person.
- Adds optional overlay of speed (v) and vertical velocity (vy).
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------
# Visualization helper
# -----------------------
def put_text(img, text, org, scale=0.6, thickness=2, bg=True, color=(0,255,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg:
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness + 2)
        x, y = org
        cv2.rectangle(img, (x, y - h - 6), (x + w + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


# -----------------------
# Geometry helpers
# -----------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1.0, float(area_a + area_b - inter))


def inter_over_ball_area(ball_box, person_box):
    bx1, by1, bx2, by2 = ball_box
    px1, py1, px2, py2 = person_box
    inter_x1 = max(bx1, px1); inter_y1 = max(by1, py1)
    inter_x2 = min(bx2, px2); inter_y2 = min(by2, py2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    ball_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    return inter / ball_area


def expand_box(b, ratio):
    x1, y1, x2, y2 = b
    w = x2 - x1; h = y2 - y1
    dx = w * ratio; dy = h * ratio
    return int(x1 - dx), int(y1 - dy), int(x2 + dx), int(y2 + dy)


def center_in_hand_zone(cx, cy, person_box, hand_ratio):
    x1, y1, x2, y2 = person_box
    if not (x1 <= cx <= x2 and y1 <= cy <= y2):
        return False
    h = y2 - y1
    hand_y = y1 + h * (1.0 - hand_ratio)
    return cy >= hand_y


def point_to_box_distance_axes(cx, cy, b):
    x1, y1, x2, y2 = b
    dx = 0.0
    if cx < x1: dx = x1 - cx
    elif cx > x2: dx = cx - x2
    dy = 0.0
    if cy < y1: dy = y1 - cy
    elif cy > y2: dy = cy - y2
    return dx, dy


# -----------------------
# Apex / flip-lite helper
# -----------------------
def compute_apex_flip(history, apex_window, min_span, fps_dt):
    """history: deque of (t, x, y) newest at right"""
    if len(history) < 3:
        return False, 0.0, 0.0, 0.0
    K = min(apex_window, len(history))
    pts = list(history)[-K:]
    ys = [p[2] for p in pts]
    xs = [p[1] for p in pts]

    vx = (xs[-1] - xs[-2]) / fps_dt
    vy = (ys[-1] - ys[-2]) / fps_dt
    speed = (vx**2 + vy**2) ** 0.5

    idx_min = int(np.argmin(ys))
    flip_lite = False
    if 0 < idx_min < K-1:
        up_span = ys[0] - ys[idx_min]
        down_span = ys[-1] - ys[idx_min]
        flip_lite = (up_span > min_span) and (down_span > min_span)

    vy_vals = []
    for i in range(1, K):
        vy_vals.append((ys[i] - ys[i-1]) / fps_dt)
    spk = max(abs(v) for v in vy_vals) if vy_vals else 0.0
    return flip_lite, vy, speed, spk


# -----------------------
# Public callable feature
# -----------------------
def detect_throw_for_targets(frame, targets, person_boxes, fps, state, draw=True, params=None):
    DEBUG_THROW = False  # set True to print gating details

    """
    Strict THROW decision on a per-frame basis, with temporal smoothing via `state`.

    Parameters
    ----------
    frame : np.ndarray (BGR)
    targets : list of dicts, each containing at least:
        {
            "tid": int,
            "bbox": (x1,y1,x2,y2),
            "cx": float, "cy": float,
            "speed": float, "vy": float, "spk": float,
            "flip_ok": bool,
            "cname": str,
            # optional:
            "history": deque([(t, x, y), ...]) or None,
            "conf": float
        }
    person_boxes : list of (x1,y1,x2,y2)
    fps : float
    state : dict holding per-id smoothing/cooldowns, e.g. {
        "airborne_count": defaultdict(int),
        "strict_air_count": defaultdict(int),
        "contact_cool": defaultdict(int)
    }
    draw : whether to draw overlays on frame
    params : dict overriding thresholds, e.g. {
        "throw_speed": 240.0,
        "vy_flip_thresh": 80.0,
        "apex_window": 9,
        "min_vert_span": 30.0,
        "person_contact_iou": 0.3,
        "ball_contained_frac": 0.5,
        "hand_zone_ratio": 0.7,
        "person_expand_ratio": 0.2,
        "min_air_dist_px": 45.0,
        "min_airborne_frames": 3,
        "strict_air_frames": 2,
        "contact_cooldown": 8,
        "max_speed_near_person": 1800.0,
        "enable_hand_contact_checks": True,
    }

    Returns
    -------
    results : list of dict
        [{"tid": int, "behavior": "THROW", "score": float}, ...]  for all targets that qualify.
    """
    # Defaults (mirroring original script where reasonable)
    defaults = dict(
        throw_speed=140,
        vy_flip_thresh=60,
        apex_window=9,
        min_vert_span=20,
        person_contact_iou=0.3,
        ball_contained_frac=0.5,
        hand_zone_ratio=0.7,
        person_expand_ratio=0.2,
        min_air_dist_px=45.0,
        min_airborne_frames=2,
        strict_air_frames=1,
        contact_cooldown=4,
        max_speed_near_person=1800.0,
        enable_hand_contact_checks=True,
    )
    P = defaults if params is None else {**defaults, **params}

    fps_dt = 1.0 / max(1e-6, float(fps))

    # Ensure required state dicts exist
    airborne_count = state.setdefault("airborne_count", defaultdict(int))
    strict_air_count = state.setdefault("strict_air_count", defaultdict(int))
    contact_cool   = state.setdefault("contact_cool",   defaultdict(int))

    results = []
    H, W = frame.shape[:2] if frame is not None else (0, 0)

    for t in targets:
        tid = int(t["tid"])
        x1, y1, x2, y2 = map(int, t["bbox"])
        cx = float(t["cx"]); cy = float(t["cy"])
        speed = float(t.get("speed", 0.0))
        vy    = float(t.get("vy", 0.0))
        spk   = float(t.get("spk", 0.0))
        flip_ok = bool(t.get("flip_ok", False))
        cname = str(t.get("cname", ""))

        # Optional: clamp noisy speeds near person
        in_contact = False
        strict_air_ok = True

        if P["enable_hand_contact_checks"] and person_boxes:
            for pb in person_boxes:
                pb_exp = expand_box(pb, P["person_expand_ratio"])
                # IoU (union-based)
                if iou((x1, y1, x2, y2), pb) >= P["person_contact_iou"]:
                    in_contact = True; strict_air_ok = False; break
                # containment by ball area
                if inter_over_ball_area((x1, y1, x2, y2), pb) >= P["ball_contained_frac"]:
                    in_contact = True; strict_air_ok = False; break
                # center in hand-zone
                if center_in_hand_zone(cx, cy, pb, P["hand_zone_ratio"]):
                    in_contact = True; strict_air_ok = False; break
                # inside expanded person bbox
                if (pb_exp[0] <= cx <= pb_exp[2]) and (pb_exp[1] <= cy <= pb_exp[3]):
                    in_contact = True; strict_air_ok = False; break
                # axis-wise proximity
                dx, dy = point_to_box_distance_axes(cx, cy, pb)
                if dx <= P["min_air_dist_px"] and dy <= P["min_air_dist_px"]:
                    in_contact = True; strict_air_ok = False; break

        if DEBUG_THROW:
            pass  # placeholder for future per-target debugging
        if in_contact and abs(speed) > P["max_speed_near_person"]:

            speed = P["max_speed_near_person"]

        # Airborne gating with cooldown + strict smoothing
        if in_contact:
            airborne_count[tid] = 0
            strict_air_count[tid] = 0
            contact_cool[tid] = P["contact_cooldown"]
        else:
            if contact_cool[tid] > 0:
                contact_cool[tid] -= 1
                strict_air_count[tid] = 0
                airborne_count[tid] = 0
            else:
                airborne_count[tid] = min(999, airborne_count[tid] + 1)
                strict_air_count[tid] = (strict_air_count[tid] + 1) if strict_air_ok else 0

        can_label = (
            (airborne_count[tid] >= P["min_airborne_frames"]) and
            (strict_air_count[tid] >= P["strict_air_frames"]) and
            (contact_cool[tid] == 0)
        )

        # Optional stronger apex/flip verification if history is provided
        hist = t.get("history", None)
        if hist is not None and len(hist) >= 3:
            flip_lite, vy_l, speed_l, spk_l = compute_apex_flip(hist, P["apex_window"], P["min_vert_span"], fps_dt)
            flip_ok = flip_ok or flip_lite  # upgrade if apex flip detected
            # Use latest measures if more reliable
            vy = vy if abs(vy) >= abs(vy_l) else vy_l
            spk = max(spk, spk_l)
            speed = max(speed, speed_l)

        behavior = None
        score = 0.0
        if can_label and (flip_ok) and speed >= P["throw_speed"]:
            behavior = "THROW"
            # A simple score: normalized speed factor with airborne persistence
            score = min(0.99, 0.6 + 0.001 * speed + 0.02 * strict_air_count[tid])

        if draw:
            # per-target overlay (แสดงแค่ v และ vy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255 if behavior else 200, 0), 2)
            overlay = f"v={speed:.0f} vy={vy:.0f}"
            put_text(frame, overlay, (x1, min(y2 + 18, H - 8)), scale=0.5, thickness=1)

        if behavior is not None:
            results.append({"tid": tid, "behavior": behavior, "score": float(score)})

    return results
