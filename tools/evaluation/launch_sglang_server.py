#!/usr/bin/env python3
"""Launch one SGLang HTTP server behind a supervising parent process.

The ``python -m sglang.launch_server`` entry point runs the HTTP server in the
same process that also creates the scheduler subprocesses.  In the RJob image
used by TB21, that path can initialize successfully and then hit a second
uvicorn bind (``EADDRINUSE``) during application startup.  Slime's rollout
engine avoids the issue by starting ``launch_server`` in a spawned child.  This
small standalone wrapper uses the same lifecycle for evaluation-only Pods.

The command-line arguments are the normal SGLang server arguments, e.g.::

    python tools/evaluation/launch_sglang_server.py \
      --model-path /path/to/model --host 127.0.0.1 --port 30001 --tp-size 8
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def _serve(server_args) -> None:
    try:
        launch_server(server_args)
    finally:
        # Match the official launcher: do not leave scheduler/detokenizer
        # descendants behind when uvicorn or the parent receives a signal.
        kill_process_tree(os.getpid(), include_parent=False)


def main(argv: list[str] | None = None) -> int:
    multiprocessing.set_start_method("spawn", force=True)
    server_args = prepare_server_args(list(sys.argv[1:] if argv is None else argv))
    child = multiprocessing.Process(target=_serve, args=(server_args,), name="sglang-http")
    child.start()
    print(f"[tb21-sglang-wrapper] child_pid={child.pid}", flush=True)

    def _forward_signal(signum, _frame) -> None:
        if child.is_alive():
            child.terminate()
            child.join(timeout=30)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)
    child.join()
    return child.exitcode or 0


if __name__ == "__main__":
    raise SystemExit(main())
