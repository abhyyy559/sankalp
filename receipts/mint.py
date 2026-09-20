#!/usr/bin/env python3
"""
Sankalp dose-receipt minter.

A receipt mints ONLY when the foil count and the voice check-in agree.
Each receipt commits to:
  - prev_hash   : SHA-256 of the previous receipt (hash chain)
  - ts          : mint timestamp (UTC)
  - slot counts : n_full / n_pressed / n_uncertain / confidence
  - strip_phash : perceptual hash of the source photo (replay protection)
  - voice_ok    : voice check-in verdict

Replay protection: the dHash of the source photo is compared against every
previously seen photo (Hamming distance < 10). A re-photographed strip is
NOT double-counted — it is logged as a gap event, never punished.

Crypto: Ed25519 (PyNaCl) signatures over the canonical JSON body;
SHA-256 chain. Signed with the device key from ledger/keys.json.

Raw photos auto-purge after minting (--purge-source, default on).
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import qrcode
from nacl.signing import SigningKey

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger"
PHASH_DUP_THRESHOLD = 10  # Hamming distance below this = same photo


def dhash(image_path):
    """64-bit difference hash of an image, as a hex string."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def load_key():
    kf = LEDGER / "keys.json"
    if not kf.exists():
        sys.exit("keys.json missing — run receipts/keygen.py first")
    seed = bytes.fromhex(json.loads(kf.read_text())["private_seed_hex"])
    return SigningKey(seed)


def last_hash():
    rf = LEDGER / "receipts.jsonl"
    if not rf.exists():
        return "GENESIS"
    lines = [ln for ln in rf.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1])["hash"] if lines else "GENESIS"


def seen_phashes():
    sf = LEDGER / "seen.json"
    return json.loads(sf.read_text()) if sf.exists() else {}


def log_gap(event):
    gf = LEDGER / "gaps.jsonl"
    with open(gf, "a") as f:
        f.write(json.dumps(event) + "\n")


def mint_qr(receipt):
    """Mint a QR PNG encoding the signed receipt summary; verify it scans.

    OpenCV's QRCodeDetector is occasionally flaky, so the decode is
    retried (plus one upscaled attempt) before giving up — a failure here
    raises instead of leaving a QR that cannot be scanned.
    """
    payload = json.dumps({
        "v": 1, "app": "sankalp",
        "hash": receipt["hash"],
        "sig": receipt["sig"],
        "ts": receipt["body"]["ts"],
        "dose": f"{receipt['body']['n_pressed']}/{receipt['body']['n_slots']}",
    }, separators=(",", ":"))
    qr_dir = LEDGER / "qr"
    qr_dir.mkdir(parents=True, exist_ok=True)
    path = qr_dir / f"{receipt['hash'][:12]}.png"
    qrcode.make(payload, box_size=10, border=2).save(str(path))
    # round-trip: prove the minted QR actually decodes. zxing-cpp is the
    # primary check (close to phone decoders); OpenCV's QRCodeDetector is
    # the fallback — it is known-flaky on some valid symbols.
    decoded = ""
    try:
        import zxingcpp
        res = zxingcpp.read_barcodes(cv2.imread(str(path)))
        decoded = res[0].text if res else ""
    except ImportError:
        pass
    if not decoded:
        det = cv2.QRCodeDetector()
        for attempt in range(3):
            img = cv2.imread(str(path))
            if attempt == 2:  # last resort: upscale before decoding
                img = cv2.resize(img, None, fx=2, fy=2,
                                 interpolation=cv2.INTER_CUBIC)
            decoded, _, _ = det.detectAndDecode(img)
            if decoded:
                break
    if not decoded or json.loads(decoded)["hash"] != receipt["hash"]:
        path.unlink(missing_ok=True)
        raise RuntimeError("QR round-trip decode failed after retries")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", required=True,
                    help="strip count JSON (from pipeline)")
    ap.add_argument("--photo", required=True, help="source strip photo")
    ap.add_argument("--voice", required=True, help="voice check-in JSON")
    ap.add_argument("--demo-date",
                    help="illustrative backdate YYYY-MM-DD (demo history only)")
    ap.add_argument("--purge-source", action="store_true", default=True)
    ap.add_argument("--keep-source", dest="purge_source", action="store_false")
    args = ap.parse_args()

    count = json.loads(Path(args.count).read_text())
    voice = json.loads(Path(args.voice).read_text())
    if count.get("status") != "accepted":
        sys.exit(f"REFUSED: count status={count.get('status')} "
                 f"({count.get('refuse_reason')}) — no receipt minted.")
    if not voice.get("voice_ok"):
        sys.exit("REFUSED: voice check-in did not agree — no receipt minted.")

    # --- replay protection -------------------------------------------------
    phash = dhash(args.photo)
    for old_hash, rid in seen_phashes().items():
        if hamming(phash, old_hash) < PHASH_DUP_THRESHOLD:
            log_gap({"event": "duplicate_photo", "phash": phash,
                     "matches_receipt": rid,
                     "ts": datetime.now(timezone.utc).isoformat(),
                     "note": "same strip re-photographed; logged as gap, "
                             "NOT double-counted"})
            print(f"DUPLICATE: photo matches receipt {rid[:12]}… "
                  f"— logged as gap, not double-counted.")
            return 2

    # --- build + sign -------------------------------------------------------
    ts = (datetime.fromisoformat(args.demo_date)
          .replace(tzinfo=timezone.utc).isoformat()
          if args.demo_date
          else datetime.now(timezone.utc).isoformat())
    body = {
        "v": 1,
        "prev_hash": last_hash(),
        "ts": ts,
        "strip_id": count.get("strip_id", ""),
        "format": count.get("format", ""),
        "n_slots": count["n_slots"],
        "n_full": count["n_full"],
        "n_pressed": count["n_pressed"],
        "n_uncertain": count["n_uncertain"],
        "confidence": count["confidence"],
        "voice_ok": voice["voice_ok"],
        "voice_method": voice.get("method", ""),
        "strip_phash": phash,
        "synthetic_input": True,
        "demo_history": bool(args.demo_date),
    }
    digest = hashlib.sha256(canonical(body)).hexdigest()
    sk = load_key()
    sig = sk.sign(canonical(body)).signature.hex()
    receipt = {"body": body, "hash": digest, "sig": sig,
               "pubkey": sk.verify_key.encode().hex()}

    LEDGER.mkdir(parents=True, exist_ok=True)
    # QR first: a decode failure must not leave a half-minted receipt
    qr_path = mint_qr(receipt)
    with open(LEDGER / "receipts.jsonl", "a") as f:
        f.write(json.dumps(receipt) + "\n")
    seen = seen_phashes()
    seen[phash] = digest
    (LEDGER / "seen.json").write_text(json.dumps(seen, indent=1))

    if args.purge_source:
        Path(args.photo).unlink(missing_ok=True)
        print(f"purged source photo {args.photo}")

    print(f"minted receipt {digest[:16]}…  "
          f"pressed={body['n_pressed']}/{body['n_slots']}  qr={qr_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
