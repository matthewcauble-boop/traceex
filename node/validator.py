"""Act as a validator on a sats node (v0.6): commit a measurement, then reveal it.

    python node/validator.py --url https://<node> --token $TRACEX_ADMIN_TOKEN --address 0xVALIDATOR \\
        --learning sha256:... --eval-set sha256:... --metric "pass@1" --before 0.294 --after 0.372 --n 167 \\
        --se 0.028 --audit-checked 10 --audit-bad 0

A validator measures the learning on its own held-out data (never shared), with the base model and with the
learning, on the same items; `se` is the paired standard error of the difference. It also audits `--audit-checked`
of the learning's parents and reports how many were junk. The commit is a hash of the attestation and a random salt;
the reveal sends both once every assigned validator has committed (or the next epoch), so nobody can copy a score.
On the public testnet these messages are relayed with the operator's token until validator keys sign them.
"""
import argparse
import json
import os
import secrets
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sdk", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceex import Client  # noqa: E402
from sats import attestation_digest  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for flag in ("--url", "--address", "--learning", "--eval-set"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--token", default=os.environ.get("TRACEX_ADMIN_TOKEN"))
    ap.add_argument("--metric", default="score")
    ap.add_argument("--before", type=float, required=True)
    ap.add_argument("--after", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--se", type=float)
    ap.add_argument("--audit-checked", type=int, default=10)
    ap.add_argument("--audit-bad", type=int, default=0)
    ap.add_argument("--round", type=int)
    ap.add_argument("--state", default=".validator-salts.json", help="where the salt waits between commit and reveal")
    a = ap.parse_args(argv)
    me = Client(a.url, a.address, token=a.token)
    att = {"validator": a.address, "eval_set": a.eval_set, "metric": a.metric, "before": a.before, "after": a.after,
           "n": a.n, "audit": {"checked": a.audit_checked, "bad": a.audit_bad}}
    if a.se is not None:
        att["se"] = a.se
    key = f"{a.address.lower()}|{a.learning}|{a.round}"   # one operator may run several validators
    state = json.load(open(a.state)) if os.path.exists(a.state) else {}
    if key not in state:
        salt = secrets.token_hex(16)
        print(json.dumps(me.commit(a.learning, attestation_digest(att, salt), a.round), indent=1))
        state[key] = {"salt": salt, "attestation": att}
        json.dump(state, open(a.state, "w"), indent=1)
        print("committed; run the same command again to reveal once the others have committed")
        return
    saved = state[key]
    print(json.dumps(me.reveal(a.learning, saved["attestation"], saved["salt"], a.round), indent=1))
    del state[key]
    json.dump(state, open(a.state, "w"), indent=1)


if __name__ == "__main__":
    main()
