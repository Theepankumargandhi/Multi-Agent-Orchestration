"""Exact-label, scope-bound numerical reconciliation; not semantic contradiction inference."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator

from agentforge_sdk._numbers import calculate_numbers, format_number, parse_number
from agentforge_sdk.document_changes import Reader, _context, _document
from agentforge_sdk.evidence_review import ReviewedCitation, ReviewedRAG
from agentforge_sdk.models import Contract, SDKError, fingerprint
from agentforge_sdk.multimodal import _seal
from agentforge_sdk.spatial import MultimodalDocument, SpatialCitation

_HASH = r"^[a-f0-9]{64}$"
_ID = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$"
Unit = Literal["USD", "EUR", "GBP", "%", "scalar"]
Operation = Literal["sum", "difference", "ratio", "percent_change"]


def _label(value):
    if value != " ".join(value.split()) or any(ord(c) < 32 for c in value):
        raise ValueError("fact labels and scopes require canonical single-line text")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("fact labels and scopes require valid UTF-8") from None
    return value


class _SourceLabel(Contract):
    source_id: str = Field(pattern=_ID)
    scope: str = Field(min_length=1, max_length=200)

    _canonical = field_validator("scope")(_label)


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """Explicit caller-selected artifact and semantic scope; neither is publisher authentication."""

    source_id: str
    scope: str
    reader: Reader = field(repr=False)

    def __post_init__(self):
        _SourceLabel(source_id=self.source_id, scope=self.scope)
        _document(self.reader)


class FactQuery(Contract):
    row_label: str = Field(min_length=1, max_length=200, repr=False)
    column_label: str = Field(min_length=1, max_length=200, repr=False)
    scope: str = Field(min_length=1, max_length=200)

    _canonical = field_validator("row_label", "column_label", "scope")(_label)


class ReconciliationConfig(Contract):
    min_distinct_artifacts: int = Field(default=2, ge=1, le=16)
    require_complete_coverage: bool = True
    max_total_regions: int = Field(default=8000, ge=1, le=16_000)
    max_total_text_chars: int = Field(default=2_000_000, ge=1000, le=8_000_000)


class FactObservation(Contract):
    source_id: str = Field(pattern=_ID)
    context_fingerprint: str = Field(pattern=_HASH)
    file_sha256: str = Field(pattern=_HASH)
    status: Literal["observed", "missing", "out_of_scope", "held", "ambiguous", "unsupported_number", "unauthorized"]
    value: str | None = Field(default=None, max_length=100, repr=False)
    unit: Unit | None = None
    # [row label, column label, value], preserving ReviewedCitation wrappers.
    citations: tuple[SpatialCitation | ReviewedCitation, ...] = Field(default=(), max_length=3, repr=False)


class FactValueGroup(Contract):
    value: str = Field(max_length=100, repr=False)
    unit: Unit
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=16)


class FactReconciliation(Contract):
    protocol: Literal["agentforge-fact-reconciliation-v1"] = "agentforge-fact-reconciliation-v1"
    collection_fingerprint: str = Field(pattern=_HASH)
    query: FactQuery = Field(repr=False)
    status: Literal["agreed", "conflict", "abstained"]
    reasons: tuple[str, ...]
    observations: tuple[FactObservation, ...] = Field(min_length=1, max_length=16, repr=False)
    groups: tuple[FactValueGroup, ...] = Field(max_length=16, repr=False)
    distinct_artifacts: int = Field(ge=0, le=16)
    value: str | None = Field(default=None, max_length=100, repr=False)
    unit: Unit | None = None
    fingerprint: str = Field(pattern=_HASH)


class FactCalculation(Contract):
    protocol: Literal["agentforge-reconciled-calculation-v1"] = "agentforge-reconciled-calculation-v1"
    collection_fingerprint: str = Field(pattern=_HASH)
    operation: Operation
    fact_fingerprints: tuple[str, ...] = Field(min_length=1, max_length=16)
    status: Literal["calculated", "abstained"]
    reason: Literal["verified_facts", "unverifiable_fact", "unresolved_fact", "incompatible_units", "zero_denominator"]
    facts: tuple[FactReconciliation, ...] = Field(default=(), max_length=16, repr=False)
    value: str | None = Field(default=None, max_length=200, repr=False)
    unit: Unit | None = None
    fingerprint: str = Field(pattern=_HASH)


@dataclass(slots=True)
class _Table:
    grid: dict
    rows: dict
    columns: dict
    complete: bool


def _tables(document):
    by_table = defaultdict(list)
    for page in document.pages:
        for region in page.regions:
            if region.kind == "table_cell":
                by_table[page.page, region.table_id].append(region)
    result = []
    for cells in by_table.values():
        grid = {(r.row, r.column): r for r in cells}
        rows, columns = defaultdict(list), defaultdict(list)
        for cell in cells:
            if cell.column == 0 and cell.row > 0:
                rows[cell.text].append(cell.row)
            if cell.row == 0 and cell.column > 0:
                columns[cell.text].append(cell.column)
        height, width = max(r.row for r in cells) + 1, max(r.column for r in cells) + 1
        result.append(_Table(grid, dict(rows), dict(columns), height >= 2 and width >= 2 and len(grid) == height * width))
    return tuple(result)


class EvidenceSet:
    """Reconcile exact table labels within an explicit scope, without calling text backends.

    No majority vote, newest-version guess, unit conversion, or independent-source
    claim is made. Distinct raw file hashes only prevent aliases of one artifact
    from satisfying the configured count. Publisher/source selection is caller-owned.
    """

    def __init__(self, sources: Sequence[EvidenceSource], *, config: ReconciliationConfig | None = None):
        if (not isinstance(sources, (list, tuple)) or not 1 <= len(sources) <= 16
                or any(not isinstance(s, EvidenceSource) for s in sources)):
            raise ValueError("reconciliation requires between 1 and 16 typed evidence sources")
        self._sources = tuple(sorted(sources, key=lambda s: s.source_id))
        if len({s.source_id for s in self._sources}) != len(self._sources):
            raise ValueError("evidence source IDs must be unique")
        self._config = ReconciliationConfig.model_validate((config or ReconciliationConfig()).model_dump())
        self._documents = {s.source_id: MultimodalDocument.model_validate(_document(s.reader).model_dump()) for s in self._sources}
        regions = [r for doc in self._documents.values() for p in doc.pages for r in p.regions]
        if (len(regions) > self._config.max_total_regions
                or sum(len(r.text) for r in regions) > self._config.max_total_text_chars):
            raise ValueError("evidence collection exceeds configured extraction bounds")
        self._tables = {key: _tables(doc) for key, doc in self._documents.items()}
        self._fingerprint = fingerprint(dict(config=self._config.model_dump(mode="json"), sources=[
            dict(source_id=s.source_id, scope=s.scope, context=_context(s.reader)) for s in self._sources]))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Stable source identifiers, not a count of independent publishers."""
        return tuple(source.source_id for source in self._sources)

    def without_sources(self, excluded: Sequence[str]) -> EvidenceSet:
        """Create a counterfactual view without weakening policy or mutating the original.

        Retained reader objects keep their live review authority. A smaller view
        may abstain because the original minimum-artifact policy still applies.
        This is a diagnostic capability, not authorization to ignore a source.
        """
        if (not isinstance(excluded, (list, tuple)) or not 1 <= len(excluded) <= 16
                or any(not isinstance(item, str) for item in excluded)
                or len(set(excluded)) != len(excluded)
                or not set(excluded).issubset(self.source_ids)):
            raise ValueError("source omission requires unique known identifiers")
        retained = [source for source in self._sources if source.source_id not in excluded]
        if not retained:
            raise ValueError("source omission must retain at least one source")
        return EvidenceSet(retained, config=self._config)

    def query_catalog(self) -> tuple[FactQuery, ...]:
        """Expose literal table axes, never values, only under current source authority.

        Catalog membership is discovery, not permission to use a fact: lookup must
        still check coverage, ambiguity and agreement. Overlong axes are omitted.
        """
        queries = {}
        for source in self._sources:
            if not self._authorized(source):
                raise SDKError("fact catalog requires current source authority")
            if not self._documents[source.source_id].ready:
                continue
            for table in self._tables[source.source_id]:
                if not table.complete:
                    continue
                for row in table.rows:
                    for column in table.columns:
                        if len(row) <= 200 and len(column) <= 200:
                            query = FactQuery(row_label=row, column_label=column, scope=source.scope)
                            queries[fingerprint(query.model_dump(mode="json"))] = query
        if not all(self._authorized(s) for s in self._sources):
            raise SDKError("fact catalog authority changed")
        return tuple(queries[key] for key in sorted(queries))

    @staticmethod
    def _authorized(source):
        return not isinstance(source.reader, ReviewedRAG) or source.reader.is_authorized()

    def _observe(self, source, query):
        doc = self._documents[source.source_id]
        payload = dict(source_id=source.source_id, context_fingerprint=_context(source.reader), file_sha256=doc.file_sha256)
        if source.scope != query.scope:
            return FactObservation(**payload, status="out_of_scope")
        if not self._authorized(source):
            return FactObservation(**payload, status="unauthorized")
        if not doc.ready:
            return FactObservation(**payload, status="held")
        candidates = [t for t in self._tables[source.source_id] if query.row_label in t.rows and query.column_label in t.columns]
        if not candidates:
            return FactObservation(**payload, status="missing")
        if len(candidates) != 1:
            return FactObservation(**payload, status="ambiguous")
        table = candidates[0]
        rows, columns = table.rows[query.row_label], table.columns[query.column_label]
        if not table.complete or len(rows) != 1 or len(columns) != 1:
            return FactObservation(**payload, status="ambiguous")
        row, column = rows[0], columns[0]
        cells = [table.grid[row, 0], table.grid[0, column], table.grid[row, column]]
        try:
            citations = tuple(source.reader.citation(c.region_id) for c in cells)
            if not all(source.reader.verify_citation(c) for c in citations) or not self._authorized(source):
                raise SDKError("source authorization changed")
        except SDKError:
            return FactObservation(**payload, status="unauthorized")
        parsed = parse_number(cells[-1].text)
        if parsed is None:
            return FactObservation(**payload, status="unsupported_number", citations=citations)
        value, unit = parsed
        return FactObservation(**payload, status="observed", value=format_number(value), unit=unit, citations=citations)

    def lookup(self, query: FactQuery) -> FactReconciliation:
        """Resolve only explicitly comparable numeric cells; unknown evidence is not agreement."""
        query = FactQuery.model_validate(query.model_dump())
        observations = [self._observe(s, query) for s in self._sources]
        # A later citation operation can outlive/revoke an earlier review lease.
        for index, source in enumerate(self._sources):
            if source.scope == query.scope and not self._authorized(source):
                previous = observations[index]
                observations[index] = FactObservation(source_id=previous.source_id,
                    context_fingerprint=previous.context_fingerprint, file_sha256=previous.file_sha256, status="unauthorized")
        grouped = defaultdict(list)
        for observation in observations:
            if observation.status == "observed":
                grouped[observation.value, observation.unit].append(observation.source_id)
        groups = tuple(FactValueGroup(value=v, unit=u, source_ids=tuple(ids)) for (v, u), ids in sorted(grouped.items()))
        distinct = len({o.file_sha256 for o in observations if o.status == "observed"})
        states = {o.status for o in observations}
        reasons = []
        for state, reason in (("unauthorized", "unauthorized_source"), ("held", "incomplete_extraction"),
                              ("ambiguous", "ambiguous_labels"), ("unsupported_number", "unsupported_number")):
            if state in states:
                reasons.append(reason)
        if states == {"out_of_scope"}:
            reasons.append("scope_not_found")
        if "missing" in states and self._config.require_complete_coverage:
            reasons.append("incomplete_coverage")
        if distinct < self._config.min_distinct_artifacts:
            reasons.append("insufficient_distinct_artifacts")
        units = {g.unit for g in groups}
        status, value, unit = "abstained", None, None
        if len(units) > 1:
            reasons.insert(0, "incompatible_units")
        elif len(groups) > 1:
            status = "conflict"
            reasons.insert(0, "numeric_values_disagree")
        elif groups and not reasons:
            status, value, unit = "agreed", groups[0].value, groups[0].unit
            reasons = ["agreed_values"]
        return _seal(FactReconciliation, dict(collection_fingerprint=self._fingerprint, query=query,
            status=status, reasons=tuple(reasons), observations=tuple(observations), groups=groups,
            distinct_artifacts=distinct, value=value, unit=unit))

    def verify_fact(self, fact: FactReconciliation) -> bool:
        """Recompute exact label/value evidence and current authority; never call a model."""
        try:
            fact = FactReconciliation.model_validate(fact.model_dump())
            return fact == self.lookup(fact.query)
        except (ValueError, TypeError, AttributeError, KeyError, SDKError):
            return False

    def calculate(self, operation: Operation, facts: Sequence[FactReconciliation]) -> FactCalculation:
        """Run a Decimal tool only when every supplied fact currently verifies and agrees."""
        if (not isinstance(operation, str) or operation not in {"sum", "difference", "ratio", "percent_change"}
                or not isinstance(facts, (list, tuple)) or not 1 <= len(facts) <= 16
                or any(not isinstance(f, FactReconciliation) for f in facts)
                or (operation != "sum" and len(facts) != 2)):
            raise ValueError("fact calculation requires a supported operation and bounded valid arity")
        facts = tuple(FactReconciliation.model_validate(f.model_dump()) for f in facts)
        if len({fingerprint(f.query.model_dump(mode="json")) for f in facts}) != len(facts):
            raise ValueError("fact calculation cannot count the same query twice")
        payload = dict(collection_fingerprint=self._fingerprint, operation=operation,
                       fact_fingerprints=tuple(f.fingerprint for f in facts), status="abstained", reason="unverifiable_fact")
        if not all(self.verify_fact(f) for f in facts):
            return _seal(FactCalculation, payload)
        if any(f.status != "agreed" for f in facts):
            return _seal(FactCalculation, {**payload, "reason": "unresolved_fact", "facts": facts})
        units = {f.unit for f in facts}
        if len(units) != 1:
            return _seal(FactCalculation, {**payload, "reason": "incompatible_units", "facts": facts})
        values = [Decimal(f.value) for f in facts]
        if ((operation == "ratio" and values[1] == 0) or (operation == "percent_change" and values[0] == 0)):
            return _seal(FactCalculation, {**payload, "reason": "zero_denominator", "facts": facts})
        value, unit = calculate_numbers(operation, values, facts[0].unit)
        if not all(self.verify_fact(f) for f in facts):
            return _seal(FactCalculation, payload)
        return _seal(FactCalculation, {**payload, "status": "calculated", "reason": "verified_facts",
                                      "facts": facts, "value": value, "unit": unit})

    def verify_calculation(self, calculation: FactCalculation, *, facts: Sequence[FactReconciliation] | None = None) -> bool:
        """Verify a supported result with its exact input facts, including their current scope."""
        try:
            calculation = FactCalculation.model_validate(calculation.model_dump())
            # An unverifiable-input receipt deliberately withholds source facts.
            inputs = calculation.facts if facts is None else facts
            if not inputs:
                return False
            return calculation == self.calculate(calculation.operation, inputs)
        except (ValueError, TypeError, AttributeError, KeyError, SDKError):
            return False
