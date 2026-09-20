#!/bin/bash
# Sankalp end-to-end demo (one command):
#   strip photo -> foil count -> confidence gate -> Hindi voice check-in
#   -> signed receipt -> QR -> replay-protection demo -> offline verifier
#
# Every number printed is measured by running the code below. Delegate: CPU.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEMO=/tmp/sankalp_demo
rm -rf "$DEMO" "$ROOT/receipts/ledger"
mkdir -p "$DEMO/inbox"

say() { echo; echo "=== $1 ==="; }

say "1/7  foil counter on the 22-strip synthetic test set (measured now)"
python3 "$ROOT/pipeline/test_pipeline.py" 2>&1 | grep -E "^(  S|OVERALL)" | tail -24
python3 - "$ROOT" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1] + "/pipeline/metrics.json"))
print(f"measured: slot_accuracy={m['overall_slot_accuracy']} "
      f"({m['overall_slots_decided']} decided, {m['overall_slots_abstained']} abstained) "
      f"mean_latency={m['mean_latency_ms']}ms CPU "
      f"gate_refused_degraded={m['gate_refused_expected_strips']}")
EOF

say "2/7  fresh device ledger + throwaway Ed25519 keys (real app: Keystore)"
python3 "$ROOT/receipts/keygen.py"

# illustrative 14-day history: 12 backdated receipts (gap on day 8 — a gap
# is data, not a violation). Each receipt is really signed; only the dates
# are illustrative, and each receipt is labelled demo_history=true.
say "3/7  minting illustrative 14-day dose history (receipts labelled demo_history)"
i=0
for spec in S02:TB-2x7:2:7 S22:DM-4x5:4:5 S04:TB-2x7:2:7 S05:DM-4x5:4:5 \
              S06:DM-4x5:4:5 S07:DM-4x5:4:5 S08:DM-4x5:4:5 S11:HT-1x10:1:10 \
              S12:HT-1x10:1:10 S13:TB-2x7:2:7 S16:DM-4x5:4:5 S21:TB-2x7:2:7; do
  sid="${spec%%:*}"; rest="${spec#*:}"; fmt="${rest%%:*}"
  rc="${rest#*:}"; rows="${rc%%:*}"; cols="${rc##*:}"
  [ "$i" -eq 5 ] && i=$((i+1))          # skip -> gap on day 8
  day=$(date -d "$((13-i)) days ago" +%F)
  cp "$ROOT/pipeline/test_data/${sid}.png" "$DEMO/inbox/${sid}.png"
  python3 "$ROOT/pipeline/foil_counter.py" "$DEMO/inbox/${sid}.png" \
      --strip-id "$sid" --rows "$rows" --cols "$cols" --format "$fmt" \
      --json-out "$DEMO/count.json" >/dev/null
  python3 "$ROOT/voice/checkin.py" --agree --out "$DEMO/voice.json" >/dev/null
  python3 "$ROOT/receipts/mint.py" --count "$DEMO/count.json" \
      --photo "$DEMO/inbox/${sid}.png" --voice "$DEMO/voice.json" \
      --demo-date "$day" --purge-source >/dev/null \
      || { echo "MINT FAILED for $sid (demo history)"; exit 1; }
  i=$((i+1))
done
echo "minted 12 backdated receipts (1 gap day); source photos auto-purged"

say "4/7  TODAY: photo -> count -> gate -> voice agrees -> receipt mints -> QR"
cp "$ROOT/pipeline/test_data/S01.png" "$DEMO/inbox/today.png"
python3 "$ROOT/pipeline/foil_counter.py" "$DEMO/inbox/today.png" \
    --strip-id S01 --rows 2 --cols 7 --format TB-2x7 \
    --json-out "$DEMO/count.json"
python3 "$ROOT/voice/checkin.py" --agree --out "$DEMO/voice.json" | head -8
python3 "$ROOT/receipts/mint.py" --count "$DEMO/count.json" \
    --photo "$DEMO/inbox/today.png" --voice "$DEMO/voice.json" --purge-source

say "5/7  confidence gate: glare strip S09 is REFUSED, no receipt mints"
cp "$ROOT/pipeline/test_data/S09.png" "$DEMO/inbox/glare.png"
python3 "$ROOT/pipeline/foil_counter.py" "$DEMO/inbox/glare.png" \
    --strip-id S09 --rows 1 --cols 10 --format HT-1x10 \
    --json-out "$DEMO/count_bad.json"
python3 "$ROOT/receipts/mint.py" --count "$DEMO/count_bad.json" \
    --photo "$DEMO/inbox/glare.png" --voice "$DEMO/voice.json" || true

say "6/7  replay protection: photographing the same strip again"
cp "$ROOT/pipeline/test_data/S01.png" "$DEMO/inbox/today2.png"
python3 "$ROOT/receipts/mint.py" --count "$DEMO/count.json" \
    --photo "$DEMO/inbox/today2.png" --voice "$DEMO/voice.json" || true

say "7/7  provider console (Office Kit bridge): verifies every receipt offline"
python3 "$ROOT/verifier/verify.py" "$ROOT/receipts/ledger" || exit 1

echo
echo "demo complete. ledger=$ROOT/receipts/ledger  drift=$ROOT/receipts/ledger/drift.png"
