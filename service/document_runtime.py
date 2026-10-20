"""Run the local SDK parser behind an explicit, bounded Docker boundary."""

from __future__ import annotations

import base64
import json
import struct
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from agentforge_sdk.spatial import ExtractionConfig, MultimodalDocument
from code_agent.bounded_process import run_bounded


class DocumentRuntime:
    image = "agentforge-document-workbench:local"

    @contextmanager
    def _session(self, payload):
        name_id = "agentforge-doc-" + uuid4().hex
        deadline = time.monotonic() + 130

        def run(arguments, timeout=10, stdin_file=None, max_chars=4000):
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                raise RuntimeError("isolated_document_deadline")
            result = run_bounded(
                ["docker", *arguments], timeout=timeout, max_chars=max_chars, stdin_file=stdin_file
            )
            if result.returncode or result.timed_out or result.stdout_bytes > max_chars:
                raise RuntimeError("isolated_document_" + arguments[0] + "_failed")
            return result.stdout.strip()

        # Resolve the operator's tag once, then create using the immutable local image ID.
        image_id = run(["image", "inspect", self.image, "--format", "{{.Id}}"])
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            raise RuntimeError("invalid local document runtime")
        with tempfile.TemporaryDirectory(prefix="agentforge-doc-") as temporary:
            directory = Path(temporary)
            source = directory / "input.json"
            serialized = json.dumps(payload)
            if len(serialized.encode()) > 4_500_000:
                raise ValueError("document worker input budget")
            source.write_text(serialized, encoding="utf-8")
            try:
                run(
                    [
                        "create",
                        "--rm",
                        "--name",
                        name_id,
                        "--pull=never",
                        "--network=none",
                        "--ipc=none",
                        "--read-only",
                        "--user=10001:10001",
                        "--cap-drop=ALL",
                        "--security-opt=no-new-privileges",
                        "--memory=512m",
                        "--memory-swap=512m",
                        "--cpus=1",
                        "--pids-limit=64",
                        "--log-driver=none",
                        "--tmpfs=/tmp:rw,noexec,nosuid,size=96m,mode=1777",
                        image_id,
                    ]
                )
                # No host directories, Docker socket, environment files or credentials enter the worker.
                run(["start", name_id])
                # Docker cp into a read-only rootfs is rejected even for tmpfs destinations.
                # A fixed program receives bounded bytes on stdin; private content never enters argv.
                with source.open("rb") as stream:
                    run(
                        [
                            "exec",
                            "-i",
                            name_id,
                            "python",
                            "-c",
                            "import sys; from pathlib import Path; data=sys.stdin.buffer.read(4500001); "
                            'assert len(data)<=4500000; Path("/tmp/input.json").write_bytes(data)',
                        ],
                        stdin_file=stream,
                    )

                def read(filename, max_chars=1_000_000):
                    return run(
                        [
                            "exec",
                            name_id,
                            "python",
                            "-c",
                            f'import sys; from pathlib import Path; sys.stdout.buffer.write(Path("/tmp/{filename}").read_bytes())',
                        ],
                        max_chars=max_chars,
                    )

                def execute(worker):
                    # Both callers select fixed source-controlled programs, never request paths.
                    run(["exec", name_id, "python", worker], timeout=90)

                yield execute, read, image_id
            finally:
                # Killing the CLI alone is insufficient. Always remove this uniquely owned container.
                cleanup = run_bounded(["docker", "rm", "-f", name_id], timeout=10, max_chars=1000)
                if cleanup.returncode:
                    raise RuntimeError("document runtime cleanup could not be confirmed")

    def probe_cells(self, raw, probe):
        from service.document_cell_contracts import CellProbe, CellProbeResult

        return self._probe(
            raw, probe, CellProbe, CellProbeResult, "/app/document_cell_worker.py", "cells.json"
        )

    def probe_pixels(self, raw, probe):
        from service.document_cell_contracts import PixelProbe, PixelProbeResult

        return self._probe(
            raw, probe, PixelProbe, PixelProbeResult, "/app/document_pixel_worker.py", "pixels.json"
        )

    def _probe(self, raw, probe, request_type, result_type, worker, filename):
        probe = request_type.model_validate(probe.model_dump())
        if not 0 < len(raw) <= 3_000_000:
            raise ValueError("document probe input budget")
        payload = {"data": base64.b64encode(raw).decode("ascii"), "probe": probe.model_dump(mode="json")}
        with self._session(payload) as (execute, read, image_id):
            execute(worker)
            result = result_type.model_validate_json(read(filename, max_chars=600_000))
        return {"result": result.model_dump(mode="json"), "runtime_image_id": image_id}

    def extract(
        self, raw: bytes, *, name: str, suffix: str, document_id: str, version: str, ocr: bool
    ) -> dict:
        payload = {
            "data": base64.b64encode(raw).decode("ascii"),
            "name": name,
            "suffix": suffix,
            "document_id": document_id,
            "version": version,
            "ocr": ocr,
        }
        with self._session(payload) as (execute, read, image_id):
            execute("/app/document_worker.py")
            manifest = MultimodalDocument.model_validate_json(read("manifest.json"))
            expected = ExtractionConfig(
                max_file_bytes=3_000_000,
                max_pages=10,
                max_pixels=6_000_000,
                max_regions=400,
                max_text_chars=100_000,
                render_dpi=100,
            )
            if (
                manifest.config != expected
                or (manifest.ocr_backend != "none") != ocr
                or manifest.media_type
                != {
                    ".pdf": "application/pdf",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                }[suffix]
            ):
                raise ValueError("document runtime profile mismatch")
            result = {
                "manifest": manifest.model_dump(mode="json"),
                "previews": [read(f"preview-{page.page}.txt") for page in manifest.pages],
            }
            if sum(len(p) for p in result["previews"]) > 7_000_000:
                raise ValueError("preview aggregate budget")
        manifest = MultimodalDocument.model_validate_json(json.dumps(result["manifest"]))
        import hashlib

        if (
            manifest.file_sha256 != hashlib.sha256(raw).hexdigest()
            or manifest.document_id != document_id
            or manifest.version != version
            or manifest.source != name
            or len(result["previews"]) != len(manifest.pages)
        ):
            raise ValueError("document runtime binding mismatch")
        for preview in result["previews"]:
            data = base64.b64decode(preview, validate=True)
            if not data.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR") or not 33 <= len(data) <= 700_000:
                raise ValueError("invalid document preview")
            width, height = struct.unpack(">II", data[16:24])
            if not 1 <= width <= 1000 or not 1 <= height <= 1400:
                raise ValueError("invalid preview geometry")
        result["runtime_image_id"] = image_id
        return result
