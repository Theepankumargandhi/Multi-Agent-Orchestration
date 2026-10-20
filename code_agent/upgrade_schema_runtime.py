"""Attested schema extraction in the existing restricted sandbox, with no host imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from code_agent.delivery import digest, manifest
from code_agent.models import SandboxPolicy
from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_contracts import _unique_pairs, result_hash
from code_agent.upgrade_models import SOURCE_VERSION, V1_PYTHON, V2_PYTHON, fingerprint
from code_agent.upgrade_schema import SCHEMA_CONFIG, SCHEMA_PROBE, schema_campaign, validate_schema_targets


def parse_schemas(stdout, policy, version):
    if len(stdout.encode()) > 65000:
        raise ValueError("schema output exceeds its byte budget")
    payload = json.loads(stdout, object_pairs_hook=_unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite schema JSON")))
    if (not isinstance(payload, dict) or set(payload) != {"protocol", "pydantic_version", "models"}
            or payload["protocol"] != "upgrade-model-schema-v1" or payload["pydantic_version"] != version
            or not isinstance(payload["models"], list)):
        raise ValueError("schema protocol or dependency mismatch")
    rows = payload["models"]
    if any(not isinstance(row, dict) for row in rows) or [row.get("model") for row in rows] != [row.model for row in policy.targets]:
        raise ValueError("schema model membership mismatch")
    for row in rows:
        if (set(row) != {"model", "schema", "sha256"} or not isinstance(row["schema"], dict)
                or len(json.dumps(row["schema"]).encode()) > 20000 or row["sha256"] != result_hash(row["schema"])):
            raise ValueError("schema integrity mismatch")
    return {row["model"]: row["schema"] for row in rows}


def read_schemas(sandbox, policy, version):
    before = digest(manifest(sandbox.workspace))
    interpreter = V1_PYTHON if version == SOURCE_VERSION else V2_PYTHON
    expected = hashlib.sha256(Path(__file__).with_name("upgrade_schema_probe.py").read_bytes()).hexdigest()
    code = f"import hashlib;print(hashlib.sha256(open({SCHEMA_PROBE!r},'rb').read()).hexdigest())"
    identity = sandbox.run([interpreter, "-I", "-c", code])
    if identity.exit_code or identity.timed_out or identity.stdout.strip() != expected:
        raise ValueError("schema harness missing or changed; build the schema image")
    outputs = []
    for _ in range(2):
        result = sandbox.run([interpreter, "-I", SCHEMA_PROBE, "/workspace/" + SCHEMA_CONFIG, version])
        if result.exit_code or result.timed_out or digest(manifest(sandbox.workspace)) != before:
            raise ValueError("schema extraction unavailable or modified its source")
        outputs.append(parse_schemas(result.stdout, policy, version))
    if outputs[0] != outputs[1] or digest(manifest(sandbox.workspace)) != before:
        raise ValueError("schema extraction was unstable")
    return outputs[0], {"fingerprint": fingerprint(outputs[0]), "model_count": len(outputs[0]),
        "controller_commands": 3, "fresh_process_repeats": 2, "harness_sha256": expected}


def prepare_schema_campaign(repository, spec, schema_policy, input_policy):
    """Synchronous freeze/admission adapter; only disposable copies receive the policy."""
    policy = SandboxPolicy(image=spec.image, allowed_images={spec.image}, max_output_chars=100000,
        command_timeout_seconds=spec.command_timeout_seconds, allowed_test_executables={V1_PYTHON})
    with DockerSandbox(repository, policy) as sandbox:
        validate_schema_targets(sandbox.workspace.root, spec, schema_policy)
        if (sandbox.workspace.root / SCHEMA_CONFIG).exists():
            raise ValueError("reserved schema config already exists")
        sandbox.workspace.write_file(SCHEMA_CONFIG, schema_policy.model_dump_json())
        schemas, receipt = read_schemas(sandbox, schema_policy, SOURCE_VERSION)
        return schema_campaign(spec, input_policy, schema_policy, schemas), receipt
