"""Explicit local OCR: bounded Tesseract TSV, no shell, model download or provider."""

from __future__ import annotations

import csv
import io
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Protocol

from pydantic import Field

from agentforge_sdk.models import BackendError, Contract
from agentforge_sdk.spatial import Box


class OCRWord(Contract):
    text: str = Field(min_length=1, max_length=500, repr=False)
    box: Box
    confidence: float = Field(ge=0, le=1)
    line: tuple[int, int, int]


class LocalOCR(Protocol):
    name: str
    external_processing: bool

    def recognize(self, png: bytes, *, width: int, height: int) -> tuple[OCRWord, ...]: ...


@dataclass(frozen=True)
class TesseractOCR:
    """Caller-installed executable and language data; instantiate without doing I/O.

    Use a trusted binary. Local subprocess timeout/output limits are not an OS sandbox.
    The single-block profile (PSM 6) does not recover scanned table structure.
    """

    executable: str = "tesseract"
    language: str = "eng"
    timeout_seconds: int = 30
    external_processing: bool = False

    def __post_init__(self):
        if (not isinstance(self.executable, str) or not self.executable or "\x00" in self.executable
                or not isinstance(self.language, str) or not re.fullmatch(r"[a-z]{3}(?:\+[a-z]{3}){0,3}", self.language)
                or type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 120
                or self.external_processing is not False):
            raise ValueError("invalid local OCR configuration")

    @property
    def name(self) -> str:
        return "tesseract-tsv-psm6:" + self.language

    def recognize(self, png: bytes, *, width: int, height: int) -> tuple[OCRWord, ...]:
        if (not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n")
                or len(png) > 30_000_000 or type(width) is not int or type(height) is not int
                or min(width, height) < 1 or width * height > 25_000_000):
            raise BackendError("OCR raster is outside the supported bounds")
        try:
            # Files prevent unbounded capture_output memory use. stderr never reaches the caller.
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                result = subprocess.run(
                    [self.executable, "stdin", "stdout", "-l", self.language, "--psm", "6", "tsv"],
                    input=png, stdout=output, stderr=errors, timeout=self.timeout_seconds,
                    check=False, shell=False,
                )
                if result.returncode or output.tell() > 2_000_000 or errors.tell() > 100_000:
                    raise ValueError("OCR failed or exceeded its output budget")
                output.seek(0)
                data = output.read(2_000_001).decode("utf-8")
            return _parse_tsv(data, width, height)
        except (OSError, ValueError, subprocess.SubprocessError):
            raise BackendError("local OCR failed; no retry or external fallback was attempted") from None


def _parse_tsv(data: str, width: int, height: int) -> tuple[OCRWord, ...]:
    words = []
    try:
        reader = csv.DictReader(io.StringIO(data), delimiter="\t")
        for row in reader:
            if row["level"] != "5" or not row["text"].strip():
                continue
            x, y, w, h = (int(row[key]) for key in ("left", "top", "width", "height"))
            confidence = float(row["conf"]) / 100
            words.append(OCRWord(
                text=row["text"].strip(), confidence=confidence,
                box=Box(x0=x / width, y0=y / height, x1=(x + w) / width, y1=(y + h) / height),
                line=tuple(int(row[key]) for key in ("block_num", "par_num", "line_num")),
            ))
            if len(words) > 10_000:
                raise ValueError("too many OCR words")
        return tuple(words)
    except (ValueError, TypeError, KeyError, csv.Error):
        raise ValueError("unsupported OCR output") from None
