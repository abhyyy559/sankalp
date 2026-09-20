#!/usr/bin/env python3
"""
Sankalp foil slot counter (CPU reference implementation).

Pipeline: photo -> strip quad detection -> perspective rectification ->
slot candidate detection -> grid clustering -> per-slot classification
(full / pressed / uncertain) -> confidence gate (refuse rather than miscount).

Delegate: CPU (OpenCV on x86_64 Linux). No NPU/GPU used or claimed.
"""

import json
import math
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# ---- confidence gate policy -------------------------------------------------
# Refusing beats miscounting: the whole strip is refused when the fraction of
# uncertain slots exceeds the limit. A small number of uncertain slots is
# tolerated but flagged on the receipt; the patient retakes the photo.
UNCERTAIN_FRAC_LIMIT = 0.12   # refuse whole strip if uncertain slots exceed this


@dataclass
class Slot:
    id: int
    cx: float
    cy: float
    radius: float
    status: str          # full | pressed | uncertain
    confidence: float


@dataclass
class StripResult:
    strip_id: str
    status: str          # accepted | refused
    n_slots: int
    n_full: int
    n_pressed: int
    n_uncertain: int
    confidence: float     # mean slot confidence (accepted strips)
    latency_ms: float
    delegate: str
    refuse_reason: str
    slots: list


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def find_strip_quad(gray):
    """Rotated-rectangle strip detection via minAreaRect (tilt-robust).

    Cascade: precise edge detection first, then progressively more aggressive
    morphological closing for soft-focus / glare-degraded shots. A candidate
    must look like a filled rectangle (rectangularity check) and cover
    3%–75% of the frame. Returns corners ordered TL/TR/BR/BL, or None (which
    the confidence gate treats as a refusal — strip_not_found).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 1.0)
    img_area = gray.shape[0] * gray.shape[1]
    configs = [
        (40, 120, 3, 0.80),    # clean shots
        (30, 90, 25, 0.75),    # soft focus / mild glare
        (20, 60, 35, 0.68),    # heavily degraded
    ]
    for low, high, k, fill_min in configs:
        edges = cv2.Canny(blur, low, high)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        best = None
        for c in contours[:10]:
            area = cv2.contourArea(c)
            if not (0.03 * img_area < area < 0.75 * img_area):
                continue
            rrect = cv2.minAreaRect(c)
            (rw, rh) = rrect[1]
            if min(rw, rh) < 1:
                continue
            rect_area = rw * rh
            fill = area / rect_area
            if fill < fill_min:            # must look like a rectangle
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect > 12:                # strip, not a sliver
                continue
            box = cv2.boxPoints(rrect).astype(np.float32)
            if best is None or fill > best[0]:
                best = (fill, box)
        if best is not None:
            return order_points(best[1])
    return None


def rectify(gray, quad, out_w=1100):
    (tl, tr, br, bl) = quad
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_l = np.linalg.norm(bl - tl)
    h_r = np.linalg.norm(br - tr)
    out_h = max(60, int(out_w * (h_l + h_r) / (w_top + w_bot)))
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(gray, M, (out_w, out_h)), M


def _fit_axis_1d(vals, span, n_hint=None):
    """RANSAC 1-D lattice fit: returns sorted node positions.

    Tries candidate spacings from pairwise distances; scores each by
    inlier circles near nodes minus a penalty for empty nodes. Robust to
    the spurious between-dome circles that Hough detection produces.
    n_hint (from the medication profile) constrains the node count.
    """
    vals = np.asarray(sorted(vals), dtype=float)
    lo, hi = vals[0], vals[-1]
    if hi - lo < 1:
        return [float(lo)]
    if n_hint is not None and n_hint > 1:
        # the medication profile tells us the node count: only spacings
        # near (span)/(n-1) are plausible — few candidates, fast and tight
        base = (hi - lo) / (n_hint - 1)
        cand = sorted({round(base * f / 2) * 2
                       for f in (0.80, 0.85, 0.90, 0.95, 1.0,
                                 1.05, 1.10, 1.15, 1.20)})
    else:
        cand = set()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                d = vals[j] - vals[i]
                if span / 30 < d < span / 2.2:
                    cand.add(round(d / 4) * 4)     # quantize to 4px
        cand = sorted(cand)
    best = None
    for dx in cand:
        for x0 in (vals[0], vals[1] if len(vals) > 1 else vals[0]):
            n = int(round((hi - x0) / dx)) + 1
            if n < 2 or n > 24:
                continue
            if n_hint is not None and n != n_hint:
                continue
            nodes = [x0 + j * dx for j in range(n)]
            inl = sum(any(abs(v - nd) < 0.40 * dx for nd in nodes)
                      for v in vals)
            missed = sum(not any(abs(v - nd) < 0.40 * dx for v in vals)
                         for nd in nodes)
            score = inl - 1.0 * missed
            if best is None or score > best[0]:
                best = (score, nodes, dx)
    if best is None:
        if n_hint is not None:                 # fall back to even spacing
            return [float(lo + j * (hi - lo) / (n_hint - 1))
                    for j in range(n_hint)] if n_hint > 1 else [float(lo)]
        return [float(lo), float(hi)]
    # refine: re-centre each node on its inliers' median
    _, nodes, dx = best
    refined = []
    for nd in nodes:
        inl = [v for v in vals if abs(v - nd) < 0.45 * dx]
        refined.append(float(np.median(inl)) if inl else nd)
    return refined


def detect_slots(rect, rows_hint=None, cols_hint=None):
    """Slot centres via Hough circles, snapped to a square lattice.

    rows_hint/cols_hint: strip format from the patient's medication profile
    (the realistic deployment path — the COUNT still comes from the foil).
    Without hints, falls back to experimental auto-inference.

    Returns list of (cx, cy, r) — one per lattice node. Nodes with no nearby
    circle keep the interpolated lattice position (they will likely classify
    as uncertain rather than vanish silently).
    """
    h, w = rect.shape
    blur = cv2.medianBlur(rect, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(w / 16),
        param1=90, param2=30,
        minRadius=int(w / 34), maxRadius=int(w / 7))
    if circles is None:
        return []
    pts = [(float(x), float(y), float(r)) for x, y, r in circles[0]]

    # non-max suppression
    pts.sort(key=lambda t: t[2], reverse=True)
    kept = []
    for x, y, r in pts:
        if all(math.hypot(x - kx, y - ky) > 0.6 * min(r, kr)
               for kx, ky, kr in kept):
            kept.append((x, y, r))
    if len(kept) < 4:
        return []

    xs = np.array([p[0] for p in kept])
    ys = np.array([p[1] for p in kept])
    r_med = float(np.median([p[2] for p in kept]))

    if rows_hint and cols_hint:
        # radius-consistent circles anchor the lattice (dome-like only)
        rc = [(x, y) for x, y, r in kept
              if abs(r - r_med) <= 0.35 * r_med] or [(x, y) for x, y, _ in kept]
        x_centres = _fit_axis_1d([p[0] for p in rc], w, cols_hint)
        y_centres = _fit_axis_1d([p[1] for p in rc], h, rows_hint)
        dx = (x_centres[-1] - x_centres[0]) / (len(x_centres) - 1) \
            if len(x_centres) > 1 else r_med * 2.4
        n_rows, n_cols = rows_hint, cols_hint
        # guard: RANSAC fallback may return a different count; resample evenly
        if len(x_centres) != cols_hint:
            x_centres = [x_centres[0] + j * (x_centres[-1] - x_centres[0])
                         / (cols_hint - 1) for j in range(cols_hint)]
        if len(y_centres) != rows_hint:
            y_centres = [y_centres[0] + i * (y_centres[-1] - y_centres[0])
                         / (rows_hint - 1) for i in range(rows_hint)] \
                if rows_hint > 1 else [float(np.median([p[1] for p in rc]))]
    else:
        # experimental auto-inference (unreliable with spurious circles)
        nn = []
        for i in range(len(kept)):
            best = min(math.hypot(kept[i][0] - kept[j][0],
                                  kept[i][1] - kept[j][1])
                       for j in range(len(kept)) if j != i)
            nn.append(best)
        dx = max(float(np.median(nn)), w / 40)
        n_cols = max(1, min(int(round((xs.max() - xs.min()) / dx)) + 1, 20))
        n_rows = max(1, min(int(round((ys.max() - ys.min()) / dx)) + 1, 10))
        x0 = float(np.median(xs[xs < xs.min() + dx / 2]))
        y0 = float(np.median(ys[ys < ys.min() + dx / 2]))
        x_centres = [x0 + j * dx for j in range(n_cols)]
        y_centres = [y0 + i * dx for i in range(n_rows)]

    # expected dome radius from lattice geometry (robust; r_med is polluted
    # by spurious circles). Dome pitch ratio ~0.78 for these strip formats.
    dy = (y_centres[-1] - y_centres[0]) / (len(y_centres) - 1) \
        if len(y_centres) > 1 else dx
    r_exp = 0.39 * min(dx, dy)

    snap_tol = 0.40 * dx
    nodes = []
    for yc in y_centres:
        for xc in x_centres:
            # snap to the nearest circle with a plausible dome radius;
            # otherwise keep the interpolated lattice position
            best, best_d = None, snap_tol
            for x, y, r in kept:
                if abs(r - r_exp) > 0.45 * r_exp:
                    continue                       # not a dome, skip it
                dd = math.hypot(x - xc, y - yc)
                if dd < best_d:
                    best, best_d = (x, y), dd
            if best is not None:
                nodes.append((best[0], best[1], r_exp))
            else:
                nodes.append((xc, yc, r_exp))
    return nodes


def cluster_grid(cands):
    """Legacy shim: nodes from detect_slots are already lattice-ordered."""
    return [cands] if cands else []


def classify_slot(rect, cx, cy, r):
    """Translation-invariant slot classifier, calibrated on synthetic strips.

    Stage 1 (FULL): segment the dark pill as a connected blob — coherent dark
      blob of pill-like size/aspect anywhere in the crop => full. This does
      not depend on exact centering, so Hough centre bias cannot break it.
    Stage 2 (PRESSED): no pill found -> compare smoothed brightness of the
      central disk vs the surrounding ring (punched-foil sheen).
    Otherwise UNCERTAIN — the confidence gate then refuses the strip rather
    than miscounting it.
    """
    h, w = rect.shape
    side = int(r * 3.0)
    x0, x1 = max(int(cx - side / 2), 0), min(int(cx + side / 2), w)
    y0, y1 = max(int(cy - side / 2), 0), min(int(cy + side / 2), h)
    crop = rect[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return "uncertain", 0.0
    mu, sigma = float(crop.mean()), float(crop.std()) + 1e-6

    # -- stage 1: pill blob -------------------------------------------------
    # bounds are relative to r^2 (scale-invariant): a pill covers ~0.67 r^2
    mask = (crop < mu - 1.1 * sigma).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    pill_area = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (0.22 * r * r < area < 1.45 * r * r):
            continue
        bw_ = stats[i, cv2.CC_STAT_WIDTH]
        bh_ = stats[i, cv2.CC_STAT_HEIGHT]
        fill = area / (bw_ * bh_ + 1e-9)
        if fill > 0.45 and max(bw_, bh_) / (min(bw_, bh_) + 1e-9) < 2.2:
            pill_area = max(pill_area, area)
    if pill_area > 0:
        strength = pill_area / (0.67 * r * r)
        return "full", min(0.60 + 0.30 * min(strength, 1.5), 0.99)

    # -- stage 2: punched-foil sheen ----------------------------------------
    sm = cv2.GaussianBlur(crop.astype(np.uint8), (0, 0), r * 0.25)
    ch, cw = sm.shape
    yy, xx = np.mgrid[0:ch, 0:cw]
    dist = np.sqrt((xx - cw / 2) ** 2 + (yy - ch / 2) ** 2)
    in_c = sm[dist < r * 0.55]
    out_c = sm[(dist >= r * 0.8) & (dist < r * 1.4)]
    if in_c.size == 0 or out_c.size == 0:
        return "uncertain", 0.10
    sheen = (float(in_c.mean()) - float(out_c.mean())) / sigma
    if sheen > 0.35:
        return "pressed", min(0.55 + sheen / 2.5, 0.99)
    return "uncertain", 0.25


def _run(image_path, strip_id="", rows_hint=None, cols_hint=None):
    t0 = time.perf_counter()
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    quad = find_strip_quad(img)
    lat = lambda: (time.perf_counter() - t0) * 1000
    if quad is None:
        return (StripResult(strip_id, "refused", 0, 0, 0, 0, 0.0,
                            round(lat(), 1), "CPU", "strip_not_found", []),
                None, None)
    rect, M = rectify(img, quad)
    cands = detect_slots(rect, rows_hint, cols_hint)
    rows = cluster_grid(cands)

    slots, sid = [], 0
    for row in rows:
        for (cx, cy, r) in row:
            status, conf = classify_slot(rect, cx, cy, r)
            slots.append(Slot(sid, round(cx, 1), round(cy, 1), round(r, 1),
                              status, round(conf, 3)))
            sid += 1

    n = len(slots)
    n_unc = sum(1 for s in slots if s.status == "uncertain")
    n_full = sum(1 for s in slots if s.status == "full")
    n_pressed = sum(1 for s in slots if s.status == "pressed")
    unc_frac = n_unc / n if n else 1.0
    mean_conf = float(np.mean([s.confidence for s in slots])) if slots else 0.0

    reason = ""
    status = "accepted"
    if n == 0:
        status, reason = "refused", "no_slots_detected"
    elif unc_frac > UNCERTAIN_FRAC_LIMIT:
        status, reason = "refused", \
            f"uncertain_fraction {unc_frac:.2f} > {UNCERTAIN_FRAC_LIMIT}"

    return (StripResult(strip_id, status, n, n_full, n_pressed, n_unc,
                        round(mean_conf, 3), round(lat(), 1), "CPU", reason,
                        [asdict(s) for s in slots]), M, rect)


def count_strip(image_path, strip_id="", rows_hint=None, cols_hint=None):
    """Public entry point: returns StripResult only.

    rows_hint/cols_hint: strip format from the patient's medication profile.
    The slot COUNT always comes from the foil image; the hint only tells the
    counter which lattice to snap detections to."""
    res, _, _ = _run(image_path, strip_id, rows_hint, cols_hint)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--strip-id", default="")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--format", default="", dest="fmt")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    res = count_strip(args.image, strip_id=args.strip_id or
                      Path(args.image).stem,
                      rows_hint=args.rows, cols_hint=args.cols)
    doc = asdict(res)
    doc["format"] = args.fmt
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"count -> {args.json_out}: {res.status} "
              f"full={res.n_full} pressed={res.n_pressed} "
              f"uncertain={res.n_uncertain} {res.latency_ms:.0f}ms CPU")
    else:
        print(json.dumps(doc, indent=1)[:1500])


if __name__ == "__main__":
    from pathlib import Path
    main()
