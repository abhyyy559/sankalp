#!/usr/bin/env python3
"""
Sankalp provider console — the "Office Kit bridge" demo.

The phone streams receipt files (receipts.jsonl) to a laptop; this script
verifies EVERYTHING offline, without ever touching raw patient photos:
  1. each receipt's SHA-256 recomputes to its recorded hash
  2. each Ed25519 signature verifies against the device public key
  3. the hash chain is intact (prev_hash links, in order)
  4. timestamps are monotonic; no perceptual-hash appears twice
Then it prints an adherence summary and a drift curve: cumulative doses
taken vs the expected schedule, plus 14-day rolling adherence against the
patient's own baseline.

Usage:  python3 verify.py [ledger_dir]
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

EXPECTED_DOSES_PER_DAY = 1   # patient's regimen: one dose / day


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def main():
    ledger = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "receipts" / "ledger"
    rf = ledger / "receipts.jsonl"
    if not rf.exists():
        sys.exit(f"no receipts at {rf}")
    receipts = [json.loads(ln) for ln in rf.read_text().splitlines()
                if ln.strip()]
    pubkey = bytes.fromhex(
        json.loads((ledger / "keys.json").read_text())["public_hex"])
    vk = VerifyKey(pubkey)

    print(f"Sankalp provider console — verifying {len(receipts)} receipts "
          f"(offline, no patient photos involved)")
    print("-" * 72)
    prev, prev_ts, seen_ph, ok = "GENESIS", None, set(), True
    for i, r in enumerate(receipts):
        body, problems = r["body"], []
        if hashlib.sha256(canonical(body)).hexdigest() != r["hash"]:
            problems.append("hash mismatch")
        try:
            vk.verify(canonical(body), bytes.fromhex(r["sig"]))
        except BadSignatureError:
            problems.append("BAD SIGNATURE")
        if body["prev_hash"] != prev:
            problems.append("chain break")
        ts = datetime.fromisoformat(body["ts"])
        if prev_ts and ts < prev_ts:
            problems.append("timestamp went backwards")
        if body["strip_phash"] in seen_ph:
            problems.append("photo hash reused (double count!)")
        seen_ph.add(body["strip_phash"])
        prev, prev_ts = r["hash"], ts
        mark = "OK " if not problems else "FAIL"
        if problems:
            ok = False
        demo = " [demo-history]" if body.get("demo_history") else ""
        print(f"  [{mark}] #{i+1} {r['hash'][:12]}… {body['ts'][:10]} "
              f"pressed={body['n_pressed']}/{body['n_slots']} "
              f"conf={body['confidence']:.2f}{demo}"
              + (f"  !! {'; '.join(problems)}" if problems else ""))
    print("-" * 72)
    print("CHAIN VERDICT:", "ALL RECEIPTS VALID ✓" if ok else "FAILURES FOUND ✗")

    gaps = [json.loads(ln) for ln in (ledger / "gaps.jsonl").read_text()
            .splitlines()] if (ledger / "gaps.jsonl").exists() else []
    dups = sum(1 for g in gaps if g.get("event") == "duplicate_photo")
    n_demo = sum(1 for r in receipts if r["body"].get("demo_history"))
    print(f"summary: {len(receipts)} dose receipts ({n_demo} illustrative "
          f"demo-history), {dups} duplicate-photo gap events (not punished)")

    # --- drift curve vs the patient's own 14-day baseline -------------------
    days = sorted({datetime.fromisoformat(r["body"]["ts"]).date()
                   for r in receipts})
    if not days:
        return 0 if ok else 1
    day0 = days[0]
    taken = [sum(1 for r in receipts
                 if datetime.fromisoformat(r["body"]["ts"]).date()
                 <= d and r["body"]["n_pressed"] > 0)
             for d in days]
    expected = [(d - day0).days + 1 for d in days]  # 1 dose/day schedule
    # 14-day rolling adherence (against own baseline, not a quota)
    roll = []
    for k, d in enumerate(days):
        window = days[max(0, k - 13): k + 1]
        exp = (window[-1] - window[0]).days + 1
        got = sum(1 for dd in window
                  if any(datetime.fromisoformat(r["body"]["ts"]).date() == dd
                          and r["body"]["n_pressed"] > 0 for r in receipts))
        roll.append(got / exp)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    xs = [(d - day0).days for d in days]
    ax1.plot(xs, expected, "--", label="expected (1 dose/day)", color="black")
    ax1.plot(xs, taken, "o-", label="doses evidenced (signed receipts)",
             color="#1a7f37")
    ax1.set_ylabel("cumulative doses")
    ax1.legend(loc="upper left")
    ax1.set_title("Sankalp — adherence drift vs schedule "
                  "(illustrative demo history)" if n_demo else
                  "Sankalp — adherence drift vs schedule")
    ax2.bar(xs, roll, color="#9e6bde", width=0.8)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("14-day rolling adherence")
    ax2.set_xlabel(f"days since {day0.isoformat()}")
    ax2.set_ylim(0, 1.15)
    fig.tight_layout()
    out = ledger / "drift.png"
    fig.savefig(out, dpi=110)
    print(f"drift curve -> {out}")
    print(f"latest 14-day adherence: {roll[-1]:.0%} "
          f"(baseline is the patient's own history — a gap is data, "
          f"not a violation)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
