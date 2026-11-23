"""Run the seven notebooks top to bottom, in order, saving outputs in place.

    python scripts/run_notebooks.py            # all
    python scripts/run_notebooks.py 03 04      # some

Stops at the first notebook that fails and prints which cell broke. A failed run is saved as
<name>.failed.ipynb, so the notebook with complete outputs is never overwritten.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

if sys.platform == "win32":  # pyzmq (used by the notebook kernel) needs the selector loop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, DeadKernelError

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def main(selected: list[str]) -> int:
    books = sorted(NB_DIR.glob("0*.ipynb"))
    if selected:
        books = [b for b in books if any(b.name.startswith(s) for s in selected)]
    if not books:
        print(f"No notebooks matched {selected}: names start with 01 to 07.")
        return 1
    for nb_path in books:
        nb = nbformat.read(nb_path, as_version=4)
        t0 = time.time()
        print(f"running {nb_path.name} ...", flush=True)
        try:
            NotebookClient(nb, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(NB_DIR)}}).execute()
        except (CellExecutionError, DeadKernelError) as err:  # a dead kernel (often memory) counts as a failure too
            failed = nb_path.with_suffix(".failed.ipynb")  # keep the last good notebook untouched
            nbformat.write(nb, failed)
            print(f"FAILED {nb_path.name} after {time.time() - t0:.0f} s (partial run saved as {failed.name})\n{str(err)[:2000]}")
            return 1
        nbformat.write(nb, nb_path)
        print(f"ok     {nb_path.name} in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
