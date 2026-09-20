#!/usr/bin/env python3
"""
Sankalp web demo — the REAL pipeline behind a clean web UI.

Every number on every page is produced by the actual repo modules:
  pipeline/foil_counter.py  -> slot counting + confidence gate
  voice/checkin.py          -> Hindi voice check-in (STUB, labelled honestly)
  receipts/mint.py          -> Ed25519/SHA-256 receipts, dHash replay guard, QR
  verifier/verify.py        -> offline chain/signature verification + drift

No fake data, no invented numbers. Demo ledger lives in
webapp/demo_ledger/ (gitignored, throwaway keys).

Run:  python3 webapp/app.py   (serves http://127.0.0.1:5000)
"""

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, abort, redirect, render_template, request,
                   send_file, url_for)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "receipts"))

from foil_counter import _run          # noqa: E402
import mint as mint_mod                # noqa: E402

app = Flask(__name__,
            template_folder=str(HERE / "templates"),
            static_folder=str(HERE / "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # reject huge uploads

DEMO_LEDGER = HERE / "demo_ledger"
INBOX = DEMO_LEDGER / "inbox"
QR_DIR = DEMO_LEDGER / "qr"
FORMATS = {"TB-2x7": (2, 7), "DM-4x5": (4, 5), "HT-1x10": (1, 10)}

# All mint/receipt state goes to the demo ledger, never the real one.
mint_mod.LEDGER = DEMO_LEDGER

# In-memory session store: cid -> {count: dict, format: str, rid: str|None}
COUNTS = {}


# ---------------------------------------------------------------- helpers

def ensure_keys():
    DEMO_LEDGER.mkdir(parents=True, exist_ok=True)
    kf = DEMO_LEDGER / "keys.json"
    if kf.exists():
        return
    from nacl.signing import SigningKey
    sk = SigningKey.generate()
    kf.write_text(json.dumps({
        "private_seed_hex": sk.encode().hex(),
        "public_hex": sk.verify_key.encode().hex(),
        "note": "THROWAWAY DEMO KEYS (web demo) — real app uses Android Keystore",
    }, indent=1))


def annotate_with_M(src_path, res, M, out_path):
    """Slot overlay on the original photo — same drawing code as
    pipeline/test_pipeline.py::annotate (banner, per-slot circles)."""
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(src_path)
    if M is not None:
        Minv = np.linalg.inv(M)
        for p in res.slots:
            pt = cv2.perspectiveTransform(
                np.float32([[[p["cx"], p["cy"]]]]), Minv)[0][0]
            x, y = int(pt[0]), int(pt[1])
            r = max(8, int(p["radius"] / 2.2))
            color = {"full": (60, 220, 60), "pressed": (60, 160, 255),
                     "uncertain": (0, 200, 255)}[p["status"]]
            cv2.circle(img, (x, y), r, color, 3)
            cv2.putText(img,
                        f"{p['id']}:{p['status'][0]}:{p['confidence']:.2f}",
                        (x - r, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
    ok = res.status == "accepted"
    banner_h = 44 if ok else 84
    cv2.rectangle(img, (0, 0), (img.shape[1], banner_h),
                  (0, 0, 0) if ok else (30, 30, 170), -1)
    banner = (f"{res.strip_id}  {res.status.upper()}  "
              f"full={res.n_full} pressed={res.n_pressed} "
              f"uncertain={res.n_uncertain}  {res.latency_ms:.0f}ms CPU")
    cv2.putText(img, banner, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    if not ok:
        cv2.putText(img, f"REFUSED: {res.refuse_reason} - no count, no receipt",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def run_count(src_path, strip_id, fmt):
    rows, cols = FORMATS[fmt]
    res, M, _ = _run(str(src_path), strip_id=strip_id,
                     rows_hint=rows, cols_hint=cols)
    doc = asdict(res)
    doc["format"] = fmt
    return res, M, doc


def store_new_count(src_path, fmt, strip_id=None):
    """Run the real counter on an image; persist everything; return cid."""
    cid = uuid.uuid4().hex[:12]
    box = INBOX / cid
    box.mkdir(parents=True, exist_ok=True)
    dest = box / "upload.png"
    shutil.copy(str(src_path), dest)
    sid = strip_id or cid[:8].upper()
    res, M, doc = run_count(dest, sid, fmt)
    (box / "count.json").write_text(json.dumps(doc, indent=1))
    annotate_with_M(dest, res, M, box / "annotated.png")
    COUNTS[cid] = {"count": doc, "format": fmt, "rid": None}
    return cid


def read_receipts():
    rf = DEMO_LEDGER / "receipts.jsonl"
    if not rf.exists():
        return []
    return [json.loads(ln) for ln in rf.read_text().splitlines() if ln.strip()]


def run_checkin_agree(box):
    """Real voice module, operator-confirmed path (the honest stub)."""
    out = box / "voice.json"
    cp = subprocess.run(
        [sys.executable, str(REPO / "voice" / "checkin.py"),
         "--agree", "--out", str(out)],
        capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"checkin.py failed: {cp.stderr}")
    return json.loads(out.read_text())


def run_mint(box):
    """Real mint.py main() with the demo ledger patched in.

    Returns (code, receipt|None, gap|None). code 2 = duplicate photo.
    """
    count_json = box / "count.json"
    voice_json = box / "voice.json"
    photo = box / "upload.png"
    old_argv = sys.argv
    sys.argv = ["mint.py", "--count", str(count_json),
                "--photo", str(photo), "--voice", str(voice_json)]
    try:
        # NOTE: mint.main() RETURNS its exit code (0 ok, 2 duplicate) and
        # only sys.exit()s on hard refusals — handle both. A SystemExit(2)
        # from argparse is a usage error, NOT a duplicate, so track the
        # source explicitly.
        ret = mint_mod.main()
        code = ret if isinstance(ret, int) else 0
        duplicate = (code == 2)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        duplicate = False
    finally:
        sys.argv = old_argv
    if duplicate:
        gf = DEMO_LEDGER / "gaps.jsonl"
        gaps = [json.loads(ln) for ln in gf.read_text().splitlines()
                if ln.strip()] if gf.exists() else []
        return 2, None, (gaps[-1] if gaps else {})
    if code != 0:
        raise RuntimeError(f"mint.py exited {code}")
    # mint.py purged upload.png; drop the annotated demo copy too, so the
    # receipt page's "source photo purged" claim is literally true.
    ann = box / "annotated.png"
    if ann.exists():
        ann.unlink()
    receipts = read_receipts()
    return 0, receipts[-1], None


def run_verifier():
    cp = subprocess.run(
        [sys.executable, str(REPO / "verifier" / "verify.py"),
         str(DEMO_LEDGER)],
        capture_output=True, text=True)
    return cp.stdout + cp.stderr, cp.returncode


def seed_demo_history():
    """Mint a few backdated, clearly-labelled demo receipts so the
    provider console has a real chain + drift curve to verify."""
    if read_receipts():
        return
    seed = INBOX / "_seed"
    seed.mkdir(parents=True, exist_ok=True)
    base = date(2026, 9, 14)
    # NOTE: S11 is deliberately NOT seeded — the demo flow uploads S11.png
    # for the accepted-strip shot, and a seeded S11 would (correctly) trip
    # the dHash replay guard instead of minting.
    for i, sid in enumerate(["S01", "S03", "S05", "S08", "S02", "S13"]):
        gt = json.loads(
            (REPO / "pipeline" / "test_data" / f"{sid}.json").read_text())
        fmt = gt["format"]
        src = seed / f"{sid}.png"
        shutil.copy(str(REPO / "pipeline" / "test_data" / f"{sid}.png"), src)
        box = INBOX / f"seed-{sid}"
        box.mkdir(parents=True, exist_ok=True)
        photo = box / "upload.png"
        shutil.copy(str(src), photo)
        res, _, doc = run_count(photo, sid, fmt)
        if res.status != "accepted":
            continue
        (box / "count.json").write_text(json.dumps(doc, indent=1))
        voice = run_checkin_agree(box)
        voice["seed_note"] = "illustrative demo history"
        (box / "voice.json").write_text(json.dumps(voice, indent=1))
        old_argv = sys.argv
        sys.argv = ["mint.py", "--count", str(box / "count.json"),
                    "--photo", str(photo), "--voice", str(box / "voice.json"),
                    "--demo-date", (base + timedelta(days=i)).isoformat()]
        try:
            ret = mint_mod.main()
            code = ret if isinstance(ret, int) else 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        finally:
            sys.argv = old_argv
        if code != 0:
            raise RuntimeError(f"seed mint {sid} exited {code}")


# ---------------------------------------------------------------- routes

@app.get("/")
def home():
    return render_template("home.html", formats=list(FORMATS))


@app.post("/count")
def count():
    f = request.files.get("photo")
    fmt = request.form.get("format", "TB-2x7")
    if fmt not in FORMATS:
        abort(400, "unknown format")
    if not f or not f.filename:
        abort(400, "no photo uploaded")
    cid = uuid.uuid4().hex[:12]
    box = INBOX / cid
    box.mkdir(parents=True, exist_ok=True)
    dest = box / "upload.png"
    f.save(str(dest))
    if cv2.imread(str(dest)) is None:
        shutil.rmtree(box, ignore_errors=True)
        abort(400, "uploaded file is not a readable image")
    res, M, doc = run_count(dest, cid[:8].upper(), fmt)
    (box / "count.json").write_text(json.dumps(doc, indent=1))
    annotate_with_M(dest, res, M, box / "annotated.png")
    COUNTS[cid] = {"count": doc, "format": fmt, "rid": None}
    return redirect(url_for("result", cid=cid))


@app.post("/sample/<sid>")
def sample(sid):
    """One-tap demo using a real test-set image (same pipeline, same code)."""
    if sid not in {"S11", "S09"}:
        abort(404)
    gt = json.loads(
        (REPO / "pipeline" / "test_data" / f"{sid}.json").read_text())
    cid = store_new_count(REPO / "pipeline" / "test_data" / f"{sid}.png",
                          gt["format"], strip_id=sid)
    return redirect(url_for("result", cid=cid))


@app.get("/result/<cid>")
def result(cid):
    entry = COUNTS.get(cid)
    if not entry:
        abort(404)
    dup = request.args.get("dup") == "1"
    return render_template("result.html", cid=cid, c=entry["count"],
                           dup=dup)


@app.get("/img/<cid>")
def img(cid):
    if cid not in COUNTS:
        abort(404)
    p = INBOX / cid / "annotated.png"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/png")


@app.get("/voice/<cid>")
def voice(cid):
    entry = COUNTS.get(cid)
    if not entry:
        abort(404)
    if entry["count"]["status"] != "accepted":
        return redirect(url_for("result", cid=cid))
    return render_template("voice.html", cid=cid, c=entry["count"])


@app.post("/mint/<cid>")
def mint(cid):
    entry = COUNTS.get(cid)
    if not entry:
        abort(404)
    if entry["count"]["status"] != "accepted":
        return redirect(url_for("result", cid=cid))
    if entry["rid"]:
        return redirect(url_for("receipt", rid=entry["rid"]))
    box = INBOX / cid
    photo = box / "upload.png"
    if not photo.exists():
        # already minted+purged earlier in this session
        if entry["rid"]:
            return redirect(url_for("receipt", rid=entry["rid"]))
        abort(410, "photo already purged and no receipt is linked")
    run_checkin_agree(box)  # real voice module (operator-confirmed stub)
    code, receipt, gap = run_mint(box)
    if code == 2:
        return redirect(url_for("result", cid=cid, dup=1))
    rid = receipt["hash"]
    entry["rid"] = rid
    return redirect(url_for("receipt", rid=rid))


@app.get("/receipt/<rid>")
def receipt(rid):
    receipts = read_receipts()
    match = [r for r in receipts if r["hash"] == rid]
    if not match:
        # allow hash-prefix lookup
        match = [r for r in receipts if r["hash"].startswith(rid)]
    if not match:
        abort(404)
    r = match[0]
    pos = receipts.index(r) + 1
    qr_name = f"{r['hash'][:12]}.png"
    return render_template("receipt.html", r=r, pos=pos, total=len(receipts),
                           qr_name=qr_name)


@app.get("/qr/<name>")
def qr(name):
    if not name.endswith(".png") or "/" in name or "\\" in name:
        abort(404)
    p = QR_DIR / name
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/png")


@app.get("/console")
def console():
    output, code = run_verifier()  # real verifier, real ledger
    lines = [ln for ln in output.splitlines()]
    verdict = next((ln for ln in lines if "CHAIN VERDICT" in ln), "")
    ok_lines = [ln for ln in lines if "[OK" in ln or "[FAIL" in ln]
    has_drift = (DEMO_LEDGER / "drift.png").exists()
    return render_template("console.html", lines=lines, ok_lines=ok_lines,
                           verdict=verdict, ok=(code == 0),
                           has_drift=has_drift,
                           n_receipts=len(read_receipts()))


@app.get("/drift.png")
def drift():
    p = DEMO_LEDGER / "drift.png"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/png")


if __name__ == "__main__":
    ensure_keys()
    seed_demo_history()
    print("Sankalp web demo on http://127.0.0.1:5000  "
          f"(demo ledger: {DEMO_LEDGER})")
    app.run(host="127.0.0.1", port=5000, debug=False)
