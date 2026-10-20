"""Offline verifier for plain numeric claim replay bundles; summary-only stdout."""

import argparse
import json
import os
import stat

from agentforge_sdk.claim_bundle import (
    MAX_BUNDLE_BYTES,
    ClaimBundleError,
    load_claim_replay_bundle,
    verify_claim_replay_bundle,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Explicit local JSON file; never a URL or archive")
    parser.add_argument(
        "--expected-fingerprint", help="Fingerprint obtained through a trusted separate channel"
    )
    parser.add_argument(
        "--require-supported",
        action="store_true",
        help="Exit 2 if replay verifies but any numeric assertion is not supported",
    )
    args = parser.parse_args(argv)
    try:
        descriptor = os.open(args.bundle, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ClaimBundleError("regular JSON file required")
            bundle = load_claim_replay_bundle(stream.read(MAX_BUNDLE_BYTES + 1))
        result = verify_claim_replay_bundle(bundle, expected_fingerprint=args.expected_fingerprint)
    except (OSError, ValueError, ClaimBundleError):
        print(json.dumps({"replay_verified": False, "reason": "invalid_or_unavailable_bundle"}))
        return 1
    print(result.model_dump_json())
    return 2 if args.require_supported and not result.all_numeric_assertions_supported else 0


if __name__ == "__main__":
    raise SystemExit(main())
