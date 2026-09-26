"""Build leakage-checked SFT and DPO datasets from reviewed flywheel preferences."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from post_training.models import (
    ChatTurn,
    DatasetManifest,
    DPORecord,
    LeakageFinding,
    SFTRecord,
)

SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class ReviewedPreference(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=100_000)
    chosen: str = Field(min_length=1, max_length=100_000)
    rejected: str = Field(min_length=1, max_length=100_000)
    source_trace_fingerprints: list[str] = Field(min_length=1)
    provenance: str
    reviewer: str = Field(min_length=1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_prompt(value: str) -> str:
    return SPACE_RE.sub(" ", value.casefold()).strip()


def _token_ngrams(value: str, size: int = 5) -> set[tuple[str, ...]]:
    words = WORD_RE.findall(value.casefold())
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_preferences(path: Path) -> tuple[list[ReviewedPreference], int]:
    records: list[ReviewedPreference] = []
    by_prompt: dict[str, ReviewedPreference] = {}
    duplicates = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = ReviewedPreference.model_validate_json(line)
        if record.chosen.strip() == record.rejected.strip():
            raise ValueError(f"chosen and rejected responses are identical at line {line_number}")
        normalized = normalize_prompt(record.prompt)
        prior = by_prompt.get(normalized)
        if prior:
            if prior.chosen.strip() != record.chosen.strip() or prior.rejected.strip() != record.rejected.strip():
                raise ValueError(f"conflicting preference labels for equivalent prompt at line {line_number}")
            duplicates += 1
            continue
        by_prompt[normalized] = record
        records.append(record)
    if not records:
        raise ValueError(f"preference dataset contains no reviewed records: {path}")
    return records, duplicates


def _assign_splits(
    records: list[ReviewedPreference], validation_fraction: float, test_fraction: float
) -> dict[str, str]:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be non-negative and sum to less than one")
    ordered = sorted(
        records,
        key=lambda item: hashlib.sha256(normalize_prompt(item.prompt).encode()).hexdigest(),
    )
    count = len(ordered)
    if count == 1:
        validation_count = test_count = 0
    elif count == 2:
        validation_count, test_count = 1, 0
    else:
        validation_count = max(1, round(count * validation_fraction)) if validation_fraction else 0
        test_count = max(1, round(count * test_fraction)) if test_fraction else 0
        while validation_count + test_count >= count:
            if test_count >= validation_count and test_count:
                test_count -= 1
            elif validation_count:
                validation_count -= 1
    split_by_id: dict[str, str] = {}
    for index, record in enumerate(ordered):
        if index < test_count:
            split = "test"
        elif index < test_count + validation_count:
            split = "validation"
        else:
            split = "train"
        split_by_id[record.id] = split
    return split_by_id


def _protected_prompts(paths: list[Path]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    prompts: list[tuple[str, str]] = []
    fingerprints: dict[str, str] = {}
    for path in paths:
        fingerprints[str(path)] = sha256_file(path)
        payload = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid protected dataset JSON at {path}:{line_number}") from exc
            prompt = str(row.get("input") or row.get("prompt") or row.get("issue") or "").strip()
            if prompt:
                prompts.append((str(row.get("id") or row.get("instance_id") or f"{path.name}:{line_number}"), prompt))
    return prompts, fingerprints


def find_leakage(
    records: list[ReviewedPreference], protected_paths: list[Path], threshold: float = 0.85
) -> tuple[list[LeakageFinding], dict[str, str]]:
    protected, fingerprints = _protected_prompts(protected_paths)
    findings: list[LeakageFinding] = []
    for training in records:
        normalized_training = normalize_prompt(training.prompt)
        training_ngrams = _token_ngrams(training.prompt)
        for protected_id, prompt in protected:
            normalized_protected = normalize_prompt(prompt)
            if normalized_training == normalized_protected:
                findings.append(
                    LeakageFinding(
                        training_id=training.id,
                        protected_id=protected_id,
                        similarity=1.0,
                        kind="exact",
                    )
                )
                continue
            similarity = _jaccard(training_ngrams, _token_ngrams(prompt))
            if similarity >= threshold:
                findings.append(
                    LeakageFinding(
                        training_id=training.id,
                        protected_id=protected_id,
                        similarity=similarity,
                        kind="near_duplicate",
                    )
                )
    return findings, fingerprints


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")


def build_training_datasets(
    source: Path,
    output_dir: Path,
    *,
    protected_paths: list[Path] | None = None,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    leakage_threshold: float = 0.85,
) -> DatasetManifest:
    """Create portable SFT/DPO JSONL while protecting frozen evaluation prompts."""
    source = source.resolve()
    protected_paths = [path.resolve() for path in (protected_paths or [])]
    records, duplicates = load_preferences(source)
    findings, protected_hashes = find_leakage(records, protected_paths, leakage_threshold)
    if findings:
        summary = ", ".join(f"{item.training_id}->{item.protected_id}" for item in findings[:5])
        raise ValueError(f"training/evaluation contamination detected: {summary}")
    splits = _assign_splits(records, validation_fraction, test_fraction)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / "sft.jsonl"
    dpo_path = output_dir / "dpo.jsonl"
    sft_records = [
        SFTRecord(
            id=record.id,
            messages=[
                ChatTurn(role="user", content=record.prompt),
                ChatTurn(role="assistant", content=record.chosen),
            ],
            split=splits[record.id],
            source_trace_fingerprints=record.source_trace_fingerprints,
            reviewer=record.reviewer,
        )
        for record in records
    ]
    dpo_records = [
        DPORecord(
            id=record.id,
            prompt=record.prompt,
            chosen=record.chosen,
            rejected=record.rejected,
            split=splits[record.id],
            source_trace_fingerprints=record.source_trace_fingerprints,
            reviewer=record.reviewer,
        )
        for record in records
    ]
    _write_jsonl(sft_path, sft_records)
    _write_jsonl(dpo_path, dpo_records)
    split_counts = {split: sum(value == split for value in splits.values()) for split in ("train", "validation", "test")}
    unsigned = {
        "source_sha256": sha256_file(source),
        "sft_sha256": sha256_file(sft_path),
        "dpo_sha256": sha256_file(dpo_path),
        "split_counts": split_counts,
        "protected_dataset_sha256": protected_hashes,
        "leakage_threshold": leakage_threshold,
    }
    fingerprint = canonical_json_sha256(unsigned)
    manifest = DatasetManifest(
        dataset_id=f"agentforge-post-training-{fingerprint[:16]}",
        source_path=str(source),
        source_sha256=unsigned["source_sha256"],
        sft_path=sft_path.name,
        sft_sha256=unsigned["sft_sha256"],
        dpo_path=dpo_path.name,
        dpo_sha256=unsigned["dpo_sha256"],
        split_counts=split_counts,
        unique_prompts=len(records),
        duplicate_records_removed=duplicates,
        protected_dataset_sha256=protected_hashes,
        leakage_findings=[],
        leakage_threshold=leakage_threshold,
        all_records_human_reviewed=all(bool(record.reviewer.strip()) for record in records),
        manifest_fingerprint=fingerprint,
    )
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def load_and_verify_manifest(path: Path) -> DatasetManifest:
    manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    for relative, expected in (
        (manifest.sft_path, manifest.sft_sha256),
        (manifest.dpo_path, manifest.dpo_sha256),
    ):
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"dataset artifact escapes manifest directory: {relative}") from exc
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"dataset artifact integrity check failed: {relative}")
    source = Path(manifest.source_path)
    if not source.is_file() or sha256_file(source) != manifest.source_sha256:
        raise ValueError("training preference source integrity check failed")
    for protected_path, expected in manifest.protected_dataset_sha256.items():
        protected = Path(protected_path)
        if not protected.is_file() or sha256_file(protected) != expected:
            raise ValueError(f"protected evaluation dataset changed: {protected_path}")
    unsigned = {
        "source_sha256": manifest.source_sha256,
        "sft_sha256": manifest.sft_sha256,
        "dpo_sha256": manifest.dpo_sha256,
        "split_counts": manifest.split_counts,
        "protected_dataset_sha256": manifest.protected_dataset_sha256,
        "leakage_threshold": manifest.leakage_threshold,
    }
    if canonical_json_sha256(unsigned) != manifest.manifest_fingerprint:
        raise ValueError("dataset manifest fingerprint verification failed")
    if not manifest.all_records_human_reviewed:
        raise ValueError("post-training requires human-reviewed records")
    return manifest
