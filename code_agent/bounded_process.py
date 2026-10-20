"""Drain a controller-owned subprocess without accumulating unbounded output."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from typing import BinaryIO


@dataclass(frozen=True)
class BoundedResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    stdout_bytes: int
    stderr_bytes: int


class _Tail:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = bytearray()
        self.total = 0
        self.error = False

    def drain(self, stream):
        try:
            while block := stream.read(16_384):
                self.total += len(block)
                self.data.extend(block)
                if len(self.data) > self.capacity:
                    del self.data[: -self.capacity]
        except OSError:
            self.error = True
        finally:
            stream.close()


def run_bounded(argv: list[str], *, timeout: float, max_chars: int,
                stdin_file: BinaryIO | None = None) -> BoundedResult:
    """Capture only UTF-8 tails. No shell; no communicate()/unbounded byte buffers.

    The caller owns the execution boundary. Killing the Docker CLI on timeout
    does not prove that its in-container process stopped; sandbox cleanup must
    still force-remove that owned container.
    """
    if not 0 < timeout <= 600 or type(max_chars) is not int or not 1 <= max_chars <= 1_000_000:
        raise ValueError("subprocess capture limits are invalid")
    tails = [_Tail(max_chars * 4), _Tail(max_chars * 4)]
    process = subprocess.Popen(
        argv, stdin=stdin_file if stdin_file is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )
    readers = [
        threading.Thread(target=tail.drain, args=(stream,), daemon=True)
        for tail, stream in zip(tails, (process.stdout, process.stderr), strict=True)
    ]
    timed_out = False
    try:
        for reader in readers:
            reader.start()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=5)
            code = 124
        for reader in readers:
            reader.join(timeout=2)
        if any(reader.is_alive() for reader in readers) or any(tail.error for tail in tails):
            raise RuntimeError("subprocess output readers did not settle")
        return BoundedResult(
            code,
            tails[0].data.decode("utf-8", errors="replace")[-max_chars:],
            tails[1].data.decode("utf-8", errors="replace")[-max_chars:],
            timed_out,
            tails[0].total,
            tails[1].total,
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for reader, stream in zip(readers, (process.stdout, process.stderr), strict=True):
            if not reader.is_alive() and not stream.closed:
                stream.close()
