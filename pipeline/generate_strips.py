#!/usr/bin/env python3
"""
Synthetic blister-strip photo generator for Sankalp.

Generates labelled strip photos with known ground truth so the foil counter
can be measured honestly. ALL images produced here are SYNTHETIC — rendered
programmatically with numpy/OpenCV. No real patient photos, no copyrighted
material. Label files record format, per-slot ground truth, and the
augmentation seed so results are reproducible.

Formats (3):
  TB-2x7  : 2 rows x 7 cols  (14 slots) — flagship TB strip
  DM-4x5  : 4 rows x 5 cols  (20 slots) — diabetes-style card
  HT-1x10 : 1 row  x 10 cols (10 slots) — hypertension-style card

Slot states: "full" (dose present), "pressed" (dose removed).
Optional difficulty flags per strip: glare / shadow / blur / tilt.
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "test_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W, CANVAS_H = 1280, 720

FORMATS = {
    "TB-2x7": (2, 7),
    "DM-4x5": (4, 5),
    "HT-1x10": (1, 10),
}


def brushed_foil(h, w, rng):
    """Aluminium foil texture: vertical gradient + streaks + noise."""
    base = np.full((h, w), 178, dtype=np.float32)
    yy = np.linspace(0, 1, h)[:, None]
    base += (yy - 0.5) * 26.0                       # lighting falloff
    streaks = rng.normal(0, 5, (h, 1)).astype(np.float32)
    base += streaks                                   # brushed lines
    base += rng.normal(0, 4, (h, w))                  # grain
    return np.clip(base, 0, 255).astype(np.uint8)


def draw_slot(strip, cx, cy, r, state, rng):
    """Render one blister dome onto the foil strip (grayscale)."""
    # dome base: slightly brighter ring with soft highlight
    cv2.circle(strip, (cx, cy), r, 205, -1, cv2.LINE_AA)
    cv2.circle(strip, (cx, cy), r, 150, 2, cv2.LINE_AA)
    # highlight arc upper-left (light source)
    cv2.ellipse(strip, (cx, cy), (r, r), 0, 120, 220, 225, 4, cv2.LINE_AA)
    if state == "full":
        # dark pill visible under the dome
        pr = int(r * 0.52)
        cv2.ellipse(strip, (cx, cy + int(r * 0.06)), (pr, int(pr * 0.78)),
                    -12, 0, 360, 92, -1, cv2.LINE_AA)
        cv2.ellipse(strip, (cx - pr // 3, cy - pr // 4), (pr // 2, pr // 3),
                    -12, 0, 360, 130, -1, cv2.LINE_AA)   # pill sheen
        cv2.ellipse(strip, (cx, cy + int(r * 0.06)), (pr, int(pr * 0.78)),
                    -12, 0, 360, 70, 2, cv2.LINE_AA)
    else:  # pressed: foil caved in, brighter flattened centre, dimple ring
        pr = int(r * 0.55)
        cv2.ellipse(strip, (cx, cy), (pr, pr), 0, 0, 360, 196, -1, cv2.LINE_AA)
        cv2.ellipse(strip, (cx, cy), (int(pr * 0.55), int(pr * 0.55)),
                    0, 0, 360, 214, -1, cv2.LINE_AA)      # punched sheen
        cv2.ellipse(strip, (cx, cy), (pr, pr), 0, 0, 360, 168, 2, cv2.LINE_AA)
    # per-slot manufacturing variance
    noise = rng.normal(0, 3, (2 * r + 6, 2 * r + 6))
    y0, y1 = max(cy - r - 3, 0), min(cy + r + 3, strip.shape[0])
    x0, x1 = max(cx - r - 3, 0), min(cx + r + 3, strip.shape[1])
    patch = strip[y0:y1, x0:x1].astype(np.float32)
    patch += noise[: y1 - y0, : x1 - x0]
    strip[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)


def make_strip(strip_id, fmt, n_pressed, flags, seed, out_dir=OUT_DIR):
    rng = np.random.default_rng(seed)
    rows, cols = FORMATS[fmt]

    # upright strip render
    margin, gap = 34, 26
    slot_r = 46 if fmt != "HT-1x10" else 52
    sw = cols * (2 * slot_r) + (cols - 1) * gap + 2 * margin
    sh = rows * (2 * slot_r) + (rows - 1) * gap + 2 * margin
    strip = brushed_foil(sh, sw, rng)
    cv2.rectangle(strip, (0, 0), (sw - 1, sh - 1), 120, 3)  # cut edge

    # choose which slots are pressed (dose removed), rest full
    idx = rng.choice(rows * cols, size=n_pressed, replace=False)
    states, centers = [], []
    k = 0
    for rr in range(rows):
        for cc in range(cols):
            cx = margin + slot_r + cc * (2 * slot_r + gap)
            cy = margin + slot_r + rr * (2 * slot_r + gap)
            state = "pressed" if k in idx else "full"
            draw_slot(strip, cx, cy, slot_r, state, rng)
            states.append(state)
            centers.append([cx, cy])
            k += 1

    # difficulty overlays (still upright, before warp)
    if "shadow" in flags:  # finger-like occlusion over some slots
        sx = int(sw * rng.uniform(0.55, 0.8))
        cv2.ellipse(strip, (sx, sh // 2), (70, sh // 2 + 40), 12, 0, 360, 70, -1, cv2.LINE_AA)
        strip = cv2.GaussianBlur(strip, (0, 0), 1.2)
    if "glare" in flags:  # blown-out reflection patch
        gx, gy = int(sw * rng.uniform(0.3, 0.6)), int(sh * rng.uniform(0.3, 0.6))
        cv2.ellipse(strip, (gx, gy), (95, 70), -20, 0, 360, 250, -1, cv2.LINE_AA)
        cv2.ellipse(strip, (gx, gy), (45, 32), -20, 0, 360, 255, -1, cv2.LINE_AA)

    # fit strip onto the canvas (HT-1x10 is long)
    max_w, max_h = CANVAS_W - 140, CANVAS_H - 140
    scale = min(1.0, max_w / sw, max_h / sh)
    if scale < 1.0:
        strip = cv2.resize(strip, (int(sw * scale), int(sh * scale)),
                           interpolation=cv2.INTER_AREA)
        sh, sw = strip.shape[:2]
        slot_r = max(18, int(slot_r * scale))
        centers = [[x * scale, y * scale] for x, y in centers]

    # place strip on a tabletop canvas with rotation + perspective
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    ty, tx = np.linspace(60, 150, CANVAS_H)[:, None], np.linspace(40, 110, CANVAS_W)[None, :]
    canvas = (ty * 0.4 + tx * 0.6).astype(np.uint8)
    canvas += rng.normal(0, 5, canvas.shape).astype(np.int16).clip(0, 255).astype(np.uint8)

    ang = float(rng.uniform(-12, 12)) if "tilt" in flags else float(rng.uniform(-6, 6))
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = int(sh * sin_a + sw * cos_a), int(sh * cos_a + sw * sin_a)
    M[0, 2] += bw / 2 - sw / 2
    M[1, 2] += bh / 2 - sh / 2
    warped = cv2.warpAffine(strip, M, (bw, bh), borderValue=170)

    # perspective skew
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    j = float(bw * 0.04)
    dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(warped, P, (bw, bh), borderValue=170)
    # mask of the true (tilted) strip footprint — avoids pasting an
    # axis-aligned box edge that a real photo would never have
    msrc = np.ones((sh, sw), dtype=np.uint8) * 255
    m1 = cv2.warpAffine(msrc, M, (bw, bh), borderValue=0)
    m2 = cv2.warpPerspective(m1, P, (bw, bh), borderValue=0)
    m2 = (m2 > 127)

    # fit the warped strip onto the canvas if a steep rotation made it big
    fit = min(1.0, (CANVAS_W - 40) / bw, (CANVAS_H - 40) / bh)
    if fit < 1.0:
        warped = cv2.resize(warped, (int(bw * fit), int(bh * fit)),
                            interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(m2.astype(np.uint8), (int(bw * fit), int(bh * fit)),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        bh, bw = warped.shape[:2]

    # paste onto canvas through the footprint mask
    ox = int(rng.uniform(20, CANVAS_W - bw - 20))
    oy = int(rng.uniform(20, CANVAS_H - bh - 20))
    region = canvas[oy: oy + bh, ox: ox + bw]
    canvas[oy: oy + bh, ox: ox + bw] = np.where(m2, warped, region)

    # global lighting / focus degradations
    bright = float(rng.uniform(0.82, 1.12))
    canvas = np.clip(canvas.astype(np.float32) * bright, 0, 255).astype(np.uint8)
    if "blur" in flags:
        canvas = cv2.GaussianBlur(canvas, (7, 7), 2.0)
    else:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    canvas = np.clip(canvas.astype(np.float32)
                     + rng.normal(0, 3.5, canvas.shape), 0, 255).astype(np.uint8)

    # transform ground-truth centres through the same geometry
    pts = np.float32(centers).reshape(-1, 1, 2)
    pts = cv2.transform(pts, M).reshape(-1, 2)
    pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), P).reshape(-1, 2)
    pts *= fit
    pts[:, 0] += ox
    pts[:, 1] += oy

    img_path = out_dir / f"{strip_id}.png"
    cv2.imwrite(str(img_path), canvas)
    label = {
        "strip_id": strip_id,
        "format": fmt,
        "rows": rows,
        "cols": cols,
        "n_slots": rows * cols,
        "synthetic": True,
        "flags": flags,
        "seed": seed,
        "n_pressed": n_pressed,
        "slots": [
            {"id": i, "cx": round(float(x), 1), "cy": round(float(y), 1),
             "state": s, "radius": slot_r}
            for i, ((x, y), s) in enumerate(zip(pts, states))
        ],
    }
    with open(out_dir / f"{strip_id}.json", "w") as f:
        json.dump(label, f, indent=1)
    return img_path, label


# 12 strips: 4 per format. S09/S10 carry heavy glare+shadow → gate should refuse.
STRIP_PLAN = [
    ("S01", "TB-2x7", 2, [], 1101),
    ("S02", "TB-2x7", 5, ["tilt"], 1102),
    ("S03", "TB-2x7", 0, ["tilt"], 1103),
    ("S04", "TB-2x7", 9, [], 1104),
    ("S05", "DM-4x5", 3, [], 2201),
    ("S06", "DM-4x5", 12, ["tilt"], 2202),
    ("S07", "DM-4x5", 7, ["blur"], 2203),
    ("S08", "DM-4x5", 1, [], 2204),
    ("S09", "HT-1x10", 4, ["glare", "shadow"], 3301),   # refusal-expected
    ("S10", "HT-1x10", 6, ["glare", "blur"], 3302),     # refusal-expected
    ("S11", "HT-1x10", 2, [], 3303),
    ("S12", "HT-1x10", 8, ["tilt"], 3304),
    # extra strips: clean-to-moderate difficulty, for the demo history ledger
    ("S13", "TB-2x7", 4, [], 1105),
    ("S14", "TB-2x7", 7, ["tilt"], 1106),
    ("S15", "TB-2x7", 1, [], 1107),
    ("S16", "DM-4x5", 9, [], 2205),
    ("S17", "DM-4x5", 5, ["tilt"], 2206),
    ("S18", "DM-4x5", 15, [], 2207),
    ("S19", "HT-1x10", 5, [], 3305),
    ("S20", "HT-1x10", 3, ["tilt"], 3306),
    # S21/S22: extra distinct strips for the demo-history ledger
    ("S21", "TB-2x7", 6, ["tilt"], 1111),
    ("S22", "DM-4x5", 11, [], 2208),
]


def main():
    for strip_id, fmt, n_pressed, flags, seed in STRIP_PLAN:
        p, lab = make_strip(strip_id, fmt, n_pressed, flags, seed)
        print(f"wrote {p.name}: {fmt} pressed={n_pressed} flags={flags}")


if __name__ == "__main__":
    main()
