"""Utilidades comunes de los notebooks de Kaggle (se copian con el repositorio)."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working/proyecto-integrador")


FALLOS: list[tuple[str, str]] = []   # fallos de la prueba de humo (se acumulan para verlos todos de una vez)


def sh(cmd: str, check: bool = True, env: dict | None = None) -> int:
    """Ejecuta un comando mostrando la salida en vivo.

    * Prueba de humo (``env`` con SMOKE=1): un fallo NO detiene el notebook; se acumula y se sigue con
      las demás etapas para ver TODOS los errores en una sola ejecución.
    * Primer comando que no es de humo: si hubo fallos, se detiene y los resume (así no se gastan horas
      de GPU con un código que ya se sabe que falla).
    * Cualquier otro fallo detiene el notebook."""
    from collections import deque

    smoke = bool(env) and env.get("SMOKE") == "1"
    if not smoke and FALLOS:
        resumen = "\n\n".join(f"### {c}\n{t}" for c, t in FALLOS)
        raise RuntimeError(f"LA PRUEBA DE HUMO FALLÓ en {len(FALLOS)} etapa(s). Copie TODO este mensaje y envíelo:\n\n{resumen}")
    if not smoke and (WORK / "models/smoke").exists():
        shutil.rmtree(WORK / "models/smoke", ignore_errors=True)    # pesos de la prueba de humo: no se usan ni se entregan
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=WORK, env={**os.environ, **(env or {})},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    tail: deque[str] = deque(maxlen=25)
    for line in p.stdout:
        print(line, end="", flush=True); tail.append(line)
    rc = p.wait()
    print(f"[{(time.time() - t0) / 60:.1f} min, código {rc}]", flush=True)
    if check and rc != 0:
        if smoke:
            FALLOS.append((cmd, "".join(tail))); print(f"!! FALLO en la prueba de humo (se continúa): {cmd}", flush=True)
        else:
            raise RuntimeError(f"Falló: {cmd}  →  copie el error de arriba y envíelo para corregirlo")
    return rc


def find_dir(pattern: str, must_contain: str, require: tuple[str, ...] = ()) -> Path | None:
    """Primer directorio bajo ``pattern`` (glob recursivo) que contiene ``must_contain`` y todo lo de ``require``.
    Kaggle ha cambiado la ruta de las competiciones (/kaggle/input/<slug> → /kaggle/input/competitions/<slug>),
    por eso se busca de forma recursiva y sin depender del nombre exacto."""
    for f in sorted(glob.glob(pattern, recursive=True)):
        d = Path(f).parent
        if "proyecto-integrador" in str(d) or "/working/" in str(d):
            continue
        if all((d / r).exists() for r in require):
            return d
    return None


def setup(extra_pip: str = "") -> None:
    """Copia el repositorio a /kaggle/working, localiza los datos e instala dependencias."""
    src = next((Path(p).parent for p in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
                if "proyecto" in p or Path(p).parent.joinpath("src/project").exists()), None)
    if src is None:
        raise FileNotFoundError("No encontré el repositorio: añada el dataset 'proyecto-integrador' como entrada")
    if not WORK.exists():
        shutil.copytree(src, WORK)
    os.chdir(WORK)
    sys.path[:0] = [str(WORK / "src"), str(WORK), str(WORK / "scripts")]
    # Sartorius: carpeta con train.csv + train/ (imágenes). Se busca en cualquier ruta de /kaggle/input.
    sart = find_dir("/kaggle/input/**/train.csv", "train.csv", require=("train",))
    if sart is not None and not any(sart.glob("train/*.png")):
        sart = next((d for d in (Path(f).parent for f in sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True)))
                     if any((d / "train").glob("*.png")) and "proyecto-integrador" not in str(d)), None)
    if sart:
        os.environ["SARTORIUS_DIR"] = str(sart)
    # Jarvis: carpeta de la competición (nombre con «jarvis») o, si no, la que tenga un CSV de entrenamiento
    jar = next((Path(p) for p in glob.glob("/kaggle/input/**/*jarvis*", recursive=True)
                if Path(p).is_dir() and "proyecto-integrador" not in p), None)
    if jar is None:
        jar = next((Path(f).parent for f in glob.glob("/kaggle/input/**/*.csv", recursive=True)
                    if "proyecto-integrador" not in f and "sartorius" not in f.lower() and "/train.csv" not in f), None)
    if jar:
        os.environ["JARVIS_DIR"] = str(jar)
    if not sart and "sartorius" in os.environ.get("KAGGLE_COMP", "sartorius"):
        print("AVISO: no encontré los datos de Sartorius. Contenido de /kaggle/input:")
        sh("ls -R /kaggle/input | head -40", check=False)
    os.environ.update(PYTHONPATH=f"{WORK / 'src'}:{WORK}", PYTHONHASHSEED="42", TF_FORCE_GPU_ALLOW_GROWTH="true",
                      TF_CPP_MIN_LOG_LEVEL="2", BUDGET=os.environ.get("BUDGET", "rapido"))
    print("Repositorio:", WORK, "| Sartorius:", sart, "| Jarvis:", jar, "| Perfil:", os.environ["BUDGET"])
    sh("pip install -q -e . --no-deps --no-build-isolation")
    if extra_pip:
        # opcionales: si la instalación conjunta falla (un solo paquete que no compila rompe a todos), se
        # reintenta uno por uno; lo que no se instale se avisa abajo
        if sh(f"pip install -q {extra_pip}", check=False) != 0:
            import shlex

            for pkg in shlex.split(extra_pip):
                sh(f"pip install -q {shlex.quote(pkg)}", check=False)
        for mod in ("gensim", "fasttext", "cellpose", "segment_anything"):
            if mod in extra_pip.replace("-", "_") or (mod == "fasttext" and "fasttext" in extra_pip):
                ok = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0
                estado = "OK" if ok else "NO DISPONIBLE (se usa la alternativa y se registra como desviación)"
                print(f"[dependencia opcional] {mod}: {estado}", flush=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'SIN GPU: active GPU en Settings → Accelerator'", check=False)


def restore_from_inputs() -> None:
    """Copia a /kaggle/working los resultados de los notebooks anteriores (añadidos como entrada).
    Orden 1A → 1B → 2 y **sin sobrescribir**: si un archivo existe en dos salidas (p. ej. el modelo
    base de U-Net++ reentrenado en 1B), prevalece el de 1A, que es el evaluado en el test."""
    prevs = sorted(glob.glob("/kaggle/input/**/proyecto-integrador", recursive=True))
    prevs = sorted(prevs, key=lambda p: (0 if "1a" in p.lower() else 1 if "1b" in p.lower() else 2, p))
    for prev in map(Path, prevs):
        if not (prev / "reports/prereg_congelado.json").exists():
            continue                      # es el dataset del código (o una carpeta anidada), no una salida
        n = 0
        for sub in ("reports/tables", "reports/manifests", "models/seg", "data/processed", ".git",
                    "reports/prereg_congelado.json"):
            s = prev / sub
            files = [s] if s.is_file() else (list(s.rglob("*")) if s.is_dir() else [])
            for f in files:
                if f.is_file():
                    dst = WORK / f.relative_to(prev)
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst); n += 1
        print(f"restaurado desde {prev}: {n} archivos")


def freeze_prereg() -> None:
    """Congela el pre-registro ANTES de entrenar: commit + etiqueta git `prereg` y huella SHA-256."""
    import hashlib

    p = WORK / "configs/prereg.yaml"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    rec = WORK / "reports/prereg_congelado.json"
    if rec.exists():
        old = json.loads(rec.read_text())
        if old["sha256"] != h:
            raise RuntimeError("¡prereg.yaml cambió después de congelarlo! Registre la desviación en `desviaciones`.")
        print("Pre-registro ya congelado:", old); return
    sh('git init -q 2>/dev/null; git config user.email "estudiante@ejemplo.com"; git config user.name "Proyecto integrador"', check=False)
    sh('git add -A && git commit -qm "Pre-registro del protocolo (antes de entrenar)" && git tag -f prereg', check=False)
    commit = subprocess.run("git rev-parse HEAD", shell=True, cwd=WORK, capture_output=True, text=True).stdout.strip()
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps({"sha256": h, "commit": commit, "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    print("Pre-registro congelado:", rec.read_text())


def pack(name: str, paths: list[str], exclude: tuple[str, ...] = ()) -> None:
    """Empaqueta salidas en /kaggle/working/<name>.zip (descargable desde la pestaña Output)."""
    import fnmatch
    import zipfile

    out = Path("/kaggle/working") / f"{name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in paths:
            for f in (WORK / rel).rglob("*") if (WORK / rel).is_dir() else [WORK / rel]:
                r = str(f.relative_to(WORK))
                if f.is_file() and not any(fnmatch.fnmatch(r, e) for e in exclude):
                    z.write(f, f.relative_to(WORK.parent))
    print("Empaquetado:", out, f"{out.stat().st_size / 2**20:.1f} MB")
