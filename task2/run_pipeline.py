"""Run the Task 2 notebooks in pipeline order and save their outputs.

Machine settings come from the RAIN_* variables in the repository's .env file or the environment (see README.md).

    python task2/run_pipeline.py              # every notebook
    python task2/run_pipeline.py 03 04 07     # selected notebooks, still in pipeline order
"""
import asyncio
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TASK2 = Path(__file__).resolve().parent
NOTEBOOKS = TASK2 / "notebooks"
ORDER = ["01", "02", "05", "06", "03", "04", "07"]
NOISE = ("WARN", "log4j", "setLogLevel", "Stage ")


def load_env(path):
    """Copy KEY=VALUE lines of a .env file into the environment; variables already set take precedence."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def tidy(nb):
    """Drop Spark's log noise from the saved outputs."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.outputs:
            if out.get("name") == "stderr":
                lines = out["text"].replace("\r", "\n").splitlines(True)
                out["text"] = "".join(l for l in lines if l.strip() and not any(k in l for k in NOISE))
        cell.outputs = [o for o in cell.outputs if o.get("name") != "stderr" or o["text"].strip()]


def run(number):
    path = next(NOTEBOOKS.glob(f"{number}_*.ipynb"))
    nb = nbformat.read(path, as_version=4)
    start = time.time()
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
    finally:
        tidy(nb)
        nbformat.write(nb, path)
    print(f"{path.name}: {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    load_env(TASK2.parent / ".env")
    chosen = sys.argv[1:] or ORDER
    for number in [n for n in ORDER if n in chosen]:
        run(number)
