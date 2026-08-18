"""Tests for the read QC helpers in plassembler.utils.qc."""

import gzip
import os
import shutil
import sys
from pathlib import Path

import pytest
from loguru import logger

from src.plassembler.utils.qc import chopper, gzip_compressor_cmd, gzip_file

TEST_DATA = Path("tests/test_data")


@pytest.fixture
def captured_errors():
    """Collect ERROR messages, without the exiting sink conftest installs.

    That sink raises SystemExit, which would stop any later sink - including this
    one - from ever seeing the message. The chopper error path exits on its own,
    so dropping the sinks for the duration still exercises the real control flow.
    """
    logger.remove()
    messages = []
    logger.add(messages.append, level="ERROR")
    try:
        yield messages
    finally:
        logger.remove()
        logger.add(sys.stderr)
        logger.add(lambda _: sys.exit(1), level="ERROR")


def fake_chopper(tmp_path, monkeypatch, script):
    """Put a stub `chopper` first on PATH.

    Lets the failure paths be driven without the real binary, and without the
    test depending on which arguments a given chopper release happens to reject.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "chopper"
    stub.write_text(script)
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_gzip_compressor_cmd_prefers_bgzip(monkeypatch):
    """bgzip ships with samtools (a hard dependency) and is multithreaded."""
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bgzip")
    assert gzip_compressor_cmd(8) == ["bgzip", "-@", "8", "-c"]


def test_gzip_compressor_cmd_falls_back_to_gzip(monkeypatch):
    """A missing bgzip must degrade to serial gzip, not fail QC."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert gzip_compressor_cmd(8) == ["gzip"]


@pytest.mark.requires_tool
@pytest.mark.skipif(shutil.which("bgzip") is None, reason="bgzip not installed")
def test_gzip_file_roundtrips(tmp_path):
    """bgzip output is a valid gzip stream that python's gzip module reads back."""
    plain = tmp_path / "reads.fastq"
    payload = "".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(1000))
    plain.write_text(payload)

    out = gzip_file(plain, threads=2)

    assert out == tmp_path / "reads.fastq.gz"
    with gzip.open(out, "rt") as fh:
        assert fh.read() == payload


@pytest.mark.requires_tool
@pytest.mark.skipif(
    shutil.which("chopper") is None or shutil.which("bgzip") is None,
    reason="chopper/bgzip not installed",
)
def test_chopper_output_is_readable_gzip(tmp_path):
    """The chopper chain still produces a readable fastq.gz of filtered reads."""
    logdir = tmp_path / "logs"
    chopper(
        str(TEST_DATA / "test_long.fastq.gz"),
        str(tmp_path),
        "500",
        "9",
        True,  # gzip_flag
        "2",
        logdir,
    )
    out = tmp_path / "chopper_long_reads.fastq.gz"
    assert out.exists() and out.stat().st_size > 0
    with gzip.open(out, "rt") as fh:
        lines = fh.readlines()
    assert lines, "chopper produced no reads"
    assert len(lines) % 4 == 0, "output is not whole fastq records"
    assert lines[0].startswith("@")


@pytest.mark.requires_tool
@pytest.mark.skipif(shutil.which("chopper") is None, reason="chopper not installed")
def test_chopper_accepts_uncompressed_input(tmp_path):
    """Plain fastq input skips the decompressor and is fed to chopper directly."""
    logdir = tmp_path / "logs"
    chopper(
        str(TEST_DATA / "test_long.fastq"),
        str(tmp_path),
        "500",
        "9",
        False,  # gzip_flag
        "2",
        logdir,
    )
    out = tmp_path / "chopper_long_reads.fastq.gz"
    assert out.exists() and out.stat().st_size > 0
    with gzip.open(out, "rt") as fh:
        assert fh.readline().startswith("@")


@pytest.mark.requires_tool
@pytest.mark.skipif(shutil.which("chopper") is None, reason="chopper not installed")
def test_chopper_reports_a_failing_stage(tmp_path):
    """A stage that exits non-zero must be surfaced. Previously only the final
    process was waited on, so a failing chopper was silently swallowed (and left
    a zombie behind)."""
    logdir = tmp_path / "logs"
    with pytest.raises(SystemExit):
        # not a gzip file, so the gunzip stage fails
        chopper(
            str(TEST_DATA / "test_long.fastq"),
            str(tmp_path),
            "500",
            "9",
            True,  # gzip_flag - wrong for this input, on purpose
            "2",
            logdir,
        )


def test_chopper_failure_is_fatal(tmp_path, monkeypatch, captured_errors):
    """A chopper that exits non-zero must stop the run and say why.

    The regression: only the last process in the chain was waited on, so a dead
    chopper went unnoticed. The compressor still wrote a valid - but empty - gzip
    member, "Finished running chopper" was still logged, and the assemblers were
    handed zero reads. The real message only ever reached the logfile.
    """
    fake_chopper(
        tmp_path,
        monkeypatch,
        "#!/bin/sh\n"
        "echo \"error: unexpected argument '--trim-approach' found\" >&2\n"
        "exit 2\n",
    )
    logdir = tmp_path / "logs"

    with pytest.raises(SystemExit):
        chopper(
            str(TEST_DATA / "test_long.fastq"),
            str(tmp_path),
            "500",
            "9",
            False,  # gzip_flag
            "2",
            logdir,
        )

    message = "\n".join(captured_errors)
    assert "chopper (return code 2)" in message
    # chopper's own diagnosis has to reach the user, not just the path to it
    assert "unexpected argument '--trim-approach' found" in message
    assert str(logdir / "chopper.err") in message


def test_chopper_rejects_empty_output(tmp_path, monkeypatch, captured_errors):
    """Zero reads is fatal even when every stage exits cleanly.

    An empty fastq.gz is well formed, so nothing downstream notices until Flye or
    Raven fails on an assembly with no input.
    """
    fake_chopper(tmp_path, monkeypatch, "#!/bin/sh\ncat > /dev/null\nexit 0\n")
    logdir = tmp_path / "logs"

    with pytest.raises(SystemExit):
        chopper(
            str(TEST_DATA / "test_long.fastq"),
            str(tmp_path),
            "500",
            "9",
            False,  # gzip_flag
            "2",
            logdir,
        )

    out = tmp_path / "chopper_long_reads.fastq.gz"
    assert out.exists(), "the empty file is still written; it must just not be used"
    assert "no reads survived filtering" in "\n".join(captured_errors)


def test_chopper_missing_binary_is_fatal(tmp_path, monkeypatch, captured_errors):
    """A chopper that is not installed at all must exit, not return quietly."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    logdir = tmp_path / "logs"

    with pytest.raises(SystemExit):
        chopper(
            str(TEST_DATA / "test_long.fastq"),
            str(tmp_path),
            "500",
            "9",
            False,  # gzip_flag
            "2",
            logdir,
        )

    assert "Error with chopper" in "\n".join(captured_errors)


@pytest.mark.requires_tool
@pytest.mark.skipif(shutil.which("chopper") is None, reason="chopper not installed")
def test_chopper_surfaces_a_real_argument_error(tmp_path, captured_errors):
    """The same path against the real binary: clap rejects the value and exits 2."""
    logdir = tmp_path / "logs"

    with pytest.raises(SystemExit):
        chopper(
            str(TEST_DATA / "test_long.fastq"),
            str(tmp_path),
            "500",
            "NOT_A_NUMBER",  # min_quality clap cannot parse
            False,  # gzip_flag
            "2",
            logdir,
        )

    message = "\n".join(captured_errors)
    assert "chopper (return code 2)" in message
    assert "NOT_A_NUMBER" in message, (
        "chopper's stderr must be surfaced, not just its path"
    )
