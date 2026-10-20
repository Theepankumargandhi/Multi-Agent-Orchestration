import sys

import pytest

from code_agent.bounded_process import run_bounded


def test_large_stdout_and_stderr_are_drained_but_only_bounded_tails_retained():
    result = run_bounded(
        [
            sys.executable,
            "-I",
            "-c",
            "import os; os.write(1, b'x' * 2000000 + b'END'); os.write(2, b'y' * 2000000 + b'ERR')",
        ],
        timeout=10,
        max_chars=128,
    )
    assert result.returncode == 0 and not result.timed_out
    assert len(result.stdout) == len(result.stderr) == 128
    assert result.stdout.endswith("END") and result.stderr.endswith("ERR")
    assert result.stdout_bytes == result.stderr_bytes == 2_000_003


def test_timeout_kills_the_owned_cli_and_retains_the_available_tail():
    result = run_bounded(
        [sys.executable, "-I", "-u", "-c", "import time; print('ready'); time.sleep(20)"],
        timeout=0.5,
        max_chars=100,
    )
    assert result.returncode == 124 and result.timed_out and "ready" in result.stdout


def test_explicit_file_stdin_is_not_embedded_in_arguments_or_closed_by_runner(tmp_path):
    source = tmp_path / 'input'
    source.write_bytes(b'authored input')
    with source.open('rb') as stream:
        result = run_bounded([sys.executable, '-I', '-c', 'import sys; print(len(sys.stdin.buffer.read()))'],
                             timeout=10, max_chars=50, stdin_file=stream)
        assert not stream.closed
    assert result.returncode == 0 and result.stdout.strip() == '14'


@pytest.mark.parametrize("limit", [0, True, 1_000_001])
def test_invalid_capture_budget_never_starts_a_process(limit):
    with pytest.raises(ValueError):
        run_bounded(["must-not-execute"], timeout=1, max_chars=limit)
