"""Construye notebooks ``.ipynb`` a partir de fuentes en formato *percent*
(``notebooks/src/*.py``, compatible con jupytext) y, opcionalmente, los ejecuta.

Formato: ``# %% [markdown]`` inicia una celda de texto (líneas con prefijo ``# ``)
y ``# %%`` una celda de código.

Ejecución: si ``nbclient`` + ``ipykernel`` están instalados se usa el kernel de
Jupyter (recomendado: ``make notebooks``). Si no, ``--fallback`` ejecuta las celdas
en un único espacio de nombres de Python capturando ``stdout`` y figuras de
matplotlib; produce un notebook equivalente ejecutado de arriba abajo.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_percent(src: str) -> list[dict]:
    cells: list[dict] = []
    cur: dict | None = None
    for line in src.splitlines():
        if line.startswith("# %%"):
            if cur is not None:
                cells.append(cur)
            cur = {"cell_type": "markdown" if "[markdown]" in line else "code", "lines": []}
            continue
        if cur is None:
            continue
        if cur["cell_type"] == "markdown":
            cur["lines"].append(line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line)
        else:
            cur["lines"].append(line)
    if cur is not None:
        cells.append(cur)
    out = []
    for c in cells:
        while c["lines"] and not c["lines"][-1].strip():
            c["lines"].pop()
        text = "\n".join(c["lines"])
        src_lines = [ln + "\n" for ln in text.split("\n")]
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip("\n")
        cell = {"cell_type": c["cell_type"], "metadata": {}, "source": src_lines}
        if c["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return out


def make_nb(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}},
        "nbformat": 4, "nbformat_minor": 4}


def execute_fallback(nb: dict, cwd: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns: dict = {"__name__": "__main__"}
    old = Path.cwd(); os.chdir(cwd)
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
    n = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            n += 1
            code = "".join(cell["source"])
            buf = io.StringIO(); outputs = []
            try:
                import ast

                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    mod = ast.parse(code)
                    last = mod.body.pop() if mod.body and isinstance(mod.body[-1], ast.Expr) else None
                    exec(compile(mod, f"<cell {n}>", "exec"), ns)
                    value = eval(compile(ast.Expression(last.value), f"<cell {n}>", "eval"), ns) if last else None
            except Exception:
                tb = traceback.format_exc()
                outputs.append({"output_type": "error", "ename": "Error", "evalue": tb.splitlines()[-1],
                                "traceback": tb.splitlines()})
                cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": buf.getvalue()}]
                                   if buf.getvalue() else []) + outputs
                cell["execution_count"] = n
                raise RuntimeError(f"Celda {n} falló:\n{tb}") from None
            if buf.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            if value is not None and not (hasattr(value, "__module__") and "matplotlib" in str(type(value))):
                data = {"text/plain": repr(value)}
                if hasattr(value, "_repr_html_"):
                    data["text/html"] = value._repr_html_()
                outputs.append({"output_type": "execute_result", "execution_count": n, "metadata": {}, "data": data})
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO()
                fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                                         "text/plain": "<Figure>"}})
            plt.close("all")
            cell["outputs"] = outputs; cell["execution_count"] = n
    finally:
        os.chdir(old)
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="ejecutor sin kernel de Jupyter")
    ap.add_argument("--out", default=str(ROOT / "notebooks"))
    a = ap.parse_args()
    srcs = [Path(s) for s in a.sources] or sorted((ROOT / "notebooks/src").glob("*.py"))
    for s in srcs:
        nb = make_nb(parse_percent(s.read_text(encoding="utf-8")))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        out = Path(a.out) / (s.stem + ".ipynb")
        if a.execute:
            if a.fallback:
                nb = execute_fallback(nb, ROOT / "notebooks")
            else:
                import nbformat
                from nbclient import NotebookClient

                for c in nb["cells"]:                       # nbclient exige `source` como str (make_nb lo guarda como lista de líneas)
                    if isinstance(c.get("source"), list):
                        c["source"] = "".join(c["source"])
                node = nbformat.from_dict(nb)
                # allow_errors: una celda que falla no aborta los demás notebooks; el error queda visible en su salida.
                NotebookClient(node, timeout=3600, kernel_name="python3", allow_errors=True,
                               resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
                for i, c in enumerate(node["cells"]):          # ERRORES_CELDAS
                    for o in c.get("outputs", []):
                        if o.get("output_type") == "error":
                            print(f"!! {s.name}: celda {i} con error {o.get('ename')}: {str(o.get('evalue'))[:200]}", flush=True)
                nb = node
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("escrito", out)


if __name__ == "__main__":
    main()
