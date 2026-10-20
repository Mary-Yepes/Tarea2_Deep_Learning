"""Comando único de reproducción (`make reproduce`). Registra entorno, semillas,
hardware, tiempo total y emisiones (codecarbon, si está instalado)."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project.config import CFG, environment_report, git_commit  # noqa: E402
from project.efficiency import carbon_tracker  # noqa: E402

STEPS = ["make test", "make seg", "make nlp", "make train", "make notebooks", "make book"]

if __name__ == "__main__":
    log = {"git_commit": git_commit(), "semillas": list(CFG.seeds_eval), "entorno": environment_report(), "pasos": []}
    t0 = time.time()
    with carbon_tracker(out_dir=str(ROOT / "reports")) as em:
        for cmd in STEPS:
            s = time.time()
            r = subprocess.run(cmd, shell=True, cwd=ROOT)
            log["pasos"].append({"cmd": cmd, "segundos": round(time.time() - s, 1), "rc": r.returncode})
            if r.returncode:
                break
    log.update(segundos_totales=round(time.time() - t0, 1), emisiones=em)
    (ROOT / "reports/reproduccion.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))
    sys.exit(max(p["rc"] for p in log["pasos"]))
