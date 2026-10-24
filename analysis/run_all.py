"""Run the whole analysis pipeline in order (final numbers: 10 000 bootstrap draws for the headline estimates).
    python analysis/run_all.py            # results/tables, results/figures, results/SUMMARY.md
EFS_OUT redirects the outputs (debugging)."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [("inflation.py", ["10000", "2000"]), ("compare.py", ["10000"]), ("special.py", []), ("extras_analysis.py", []), ("revision.py", []), ("revision2.py", []), ("figures.py", []), ("report.py", [])]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name, args in STEPS:
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, name)] + args, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("%-22s exit %d  %.0fs" % (name, r.returncode, time.time() - t0), flush=True)
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-2500:])
            raise SystemExit("step failed: " + name)


if __name__ == "__main__":
    main()
