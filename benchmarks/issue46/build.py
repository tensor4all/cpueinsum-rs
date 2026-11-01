#!/usr/bin/env python3
"""Build baseline and the diagnostic patch, restoring production source on exit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "crates/cpueinsum/src/grouped.rs"


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", str(SOURCE), "Cargo.lock"], cwd=ROOT, check=True)
    original = SOURCE.read_bytes()
    metadata = {}
    try:
        for arm in ("baseline", "candidate"):
            if arm == "candidate":
                subprocess.run(["git", "apply", "--check", str(HERE / "team.patch")], cwd=ROOT, check=True)
                subprocess.run(["git", "apply", str(HERE / "team.patch")], cwd=ROOT, check=True)
            env = dict(os.environ, ISSUE46_BUILD=arm)
            command = ["nice", "-n", "5", "cargo", "bench", "--locked", "-p", "cpueinsum", "--bench", "issue46_grouped", "--no-run", "-j8", "--message-format=json"]
            p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
            (output / f"build-{arm}.jsonl").write_text(p.stdout)
            (output / f"build-{arm}.log").write_text(p.stderr)
            p.check_returncode()
            executable = None
            for line in p.stdout.splitlines():
                event = json.loads(line)
                if event.get("reason") == "compiler-artifact" and event["target"]["name"] == "issue46_grouped":
                    executable = event.get("executable")
            assert executable is not None
            shutil.copy2(executable, output / arm)
            metadata[arm] = {"command": command, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                             "binary_sha256": hashlib.sha256((output / arm).read_bytes()).hexdigest()}
    finally:
        # This script owns only its temporary patch, not arbitrary user edits.
        SOURCE.write_bytes(original)
    (output / "builds.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
