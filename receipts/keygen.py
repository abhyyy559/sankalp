#!/usr/bin/env python3
"""Generate a throwaway Ed25519 keypair for the demo ledger.

Real product: keys live in the Android Keystore / StrongBox and never leave
the phone. These demo keys exist only so the receipt chain can be exercised
end-to-end on this VM.
"""
import json
from pathlib import Path

from nacl.signing import SigningKey

OUT = Path(__file__).resolve().parent / "ledger" / "keys.json"


def main():
    sk = SigningKey.generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "private_seed_hex": sk.encode().hex(),
        "public_hex": sk.verify_key.encode().hex(),
        "note": "THROWAWAY DEMO KEYS — real app uses Android Keystore",
    }, indent=1))
    print(f"wrote {OUT}  pub={sk.verify_key.encode().hex()[:16]}…")


if __name__ == "__main__":
    main()
