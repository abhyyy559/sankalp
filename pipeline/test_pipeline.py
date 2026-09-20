#!/usr/bin/env python3
"""
Sankalp pipeline test harness.

Runs foil_counter over the 22-strip synthetic test set (3 formats),
matches each predicted slot to ground truth by nearest centre in
rectified coordinates, and measures:
  - slot-label accuracy (full/pressed) among matched slots, per format
  - grid detection rate (predicted slot count vs ground truth)
  - gate behaviour on the two refusal-expected strips (S09, S10)
  - latency per strip in ms (CPU)

Writes metrics.json. All numbers are measured by running this file.
"""

import glob
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from foil_counter import _run

HERE = Path(__file__).resolve().parent
DATA = HERE / "test_data"
REFUSAL_EXPECTED = {"S09", "S10"}   # heavy glare/shadow/blur: gate must refuse


def match_to_gt(res, M, gt):
    """Nearest-centre matching of predicted slots to GT in rect coords."""
    pts = np.float32([[s["cx"], s["cy"]] for s in gt["slots"]]).reshape(-1, 1, 2)
    gt_rect = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    used, pairs = set(), []
    spacing = float(np.median([math.hypot(
        gt_rect[i][0] - gt_rect[j][0], gt_rect[i][1] - gt_rect[j][1])
        for i in range(len(gt_rect)) for j in range(i + 1, len(gt_rect))]))
    for p in res.slots:
        best, best_d = None, 0.5 * spacing
        for gi, (gx, gy) in enumerate(gt_rect):
            if gi in used:
                continue
            d = math.hypot(p["cx"] - gx, p["cy"] - gy)
            if d < best_d:
                best, best_d = gi, d
        if best is not None:
            used.add(best)
            pairs.append((p, gt["slots"][best]))
    return pairs, spacing


def annotate(sid, res, M, gt, out_path):
    """Overlay counted slots on the original photo (for README screenshot)."""
    img = cv2.imread(str(DATA / f"{sid}.png"))
    Minv = np.linalg.inv(M)
    for p in res.slots:
        pt = cv2.perspectiveTransform(
            np.float32([[[p["cx"], p["cy"]]]]), Minv)[0][0]
        x, y = int(pt[0]), int(pt[1])
        r = max(8, int(p["radius"] / 2.2))
        color = {"full": (60, 220, 60), "pressed": (60, 160, 255),
                 "uncertain": (0, 200, 255)}[p["status"]]
        cv2.circle(img, (x, y), r, color, 3)
        cv2.putText(img, f"{p['id']}:{p['status'][0]}:{p['confidence']:.2f}",
                    (x - r, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    banner = f"{sid}  {res.status.upper()}  full={res.n_full} pressed={res.n_pressed} " \
             f"uncertain={res.n_uncertain}  {res.latency_ms:.0f}ms CPU"
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, banner, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def main():
    t_start = time.perf_counter()
    per_strip, fmt_stats = [], {}
    for p in sorted(glob.glob(str(DATA / "S*.png"))):
        sid = Path(p).stem
        gt = json.load(open(DATA / f"{sid}.json"))
        res, M, _ = _run(p, strip_id=sid,
                                   rows_hint=gt["rows"], cols_hint=gt["cols"])
        entry = {"strip_id": sid, "format": gt["format"],
                 "synthetic": True, "flags": gt["flags"],
                 "status": res.status, "refuse_reason": res.refuse_reason,
                 "latency_ms": res.latency_ms, "delegate": res.delegate,
                 "n_gt": gt["n_slots"], "n_pred": res.n_slots}
        if res.status == "accepted" and M is not None:
            pairs, _ = match_to_gt(res, M, gt)
            decided = [(p, g) for p, g in pairs if p["status"] != "uncertain"]
            correct = sum(1 for p, g in decided if p["status"] == g["state"])
            entry.update({
                "matched": len(pairs),
                "decided": len(decided),
                "correct": correct,
                # accuracy among slots the counter committed to (abstentions
                # reported separately — refusing beats miscounting)
                "slot_accuracy": round(correct / len(decided), 4)
                if decided else None,
                "abstention_rate": round(
                    (len(pairs) - len(decided)) / len(pairs), 4)
                if pairs else 0.0,
                "detection_rate": round(len(pairs) / gt["n_slots"], 4),
            })
        else:
            entry.update({"matched": 0, "correct": 0,
                          "slot_accuracy": None, "detection_rate": 0.0})
        per_strip.append(entry)
        fs = fmt_stats.setdefault(gt["format"],
                                  {"strips": 0, "matched": 0, "decided": 0,
                                   "correct": 0, "abstained": 0,
                                   "lat_ms": [], "accepted": 0, "refused": 0})
        fs["strips"] += 1
        fs["matched"] += entry["matched"]
        fs["decided"] += entry.get("decided", 0)
        fs["correct"] += entry["correct"]
        fs["abstained"] += entry["matched"] - entry.get("decided", 0)
        fs["lat_ms"].append(res.latency_ms)
        fs["refused" if res.status == "refused" else "accepted"] += 1

    formats = {}
    for fmt, fs in fmt_stats.items():
        formats[fmt] = {
            "n_strips": fs["strips"],
            "accepted": fs["accepted"], "refused": fs["refused"],
            "slot_accuracy": round(fs["correct"] / fs["decided"], 4)
            if fs["decided"] else None,
            "slots_decided": fs["decided"],
            "slots_abstained": fs["abstained"],
            "mean_latency_ms": round(float(np.mean(fs["lat_ms"])), 1),
        }
    all_matched = sum(f["matched"] for f in fmt_stats.values())
    all_decided = sum(f["decided"] for f in fmt_stats.values())
    all_correct = sum(f["correct"] for f in fmt_stats.values())
    all_abstained = sum(f["abstained"] for f in fmt_stats.values())
    all_lat = [e["latency_ms"] for e in per_strip]
    # gate check: every refusal-expected strip must be refused
    gate_ok = all(e["status"] == "refused" for e in per_strip
                  if e["strip_id"] in REFUSAL_EXPECTED)

    metrics = {
        "generated_by": "pipeline/test_pipeline.py (measured, not estimated)",
        "delegate": "CPU (opencv-python-headless on x86_64 Linux VM)",
        "test_set": "22 synthetic strips, 3 formats (ALL images synthetic; "
                    "no real patient photos)",
        "format_hint": "rows/cols passed from the strip label as the medication-"
                    "profile hint (realistic deployment path); slot states are "
                    "still counted from the foil image",
        "overall_slot_accuracy": round(all_correct / all_decided, 4)
        if all_decided else None,
        "overall_slots_decided": all_decided,
        "overall_slots_abstained": all_abstained,
        "mean_latency_ms": round(float(np.mean(all_lat)), 1),
        "gate_refused_expected_strips": gate_ok,
        "formats": formats,
        "per_strip": per_strip,
        "wall_time_s": round(time.perf_counter() - t_start, 1),
    }
    with open(HERE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)

    # README screenshot: best accepted strip (highest accuracy, then speed)
    accepted = [e for e in per_strip if e["status"] == "accepted"]
    if accepted:
        best = max(accepted, key=lambda e: (e["slot_accuracy"], -e["latency_ms"]))
        sid = best["strip_id"]
        gt = json.load(open(DATA / f"{sid}.json"))
        res, M, _ = _run(str(DATA / f"{sid}.png"), strip_id=sid)
        annotate(sid, res, M, gt, HERE / "counted_example.png")
        print(f"screenshot: counted_example.png from {sid}")

    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "per_strip"}, indent=1))
    for e in per_strip:
        print(f"  {e['strip_id']} {e['format']:8s} {e['status']:8s} "
              f"acc={e['slot_accuracy']} det={e['detection_rate']:.2f} "
              f"lat={e['latency_ms']:6.1f}ms {e['refuse_reason']}")


if __name__ == "__main__":
    main()
