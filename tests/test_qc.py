"""Tests for the read QC helpers in plassembler.utils.qc."""

import gzip
import shutil
from pathlib import Path

import pytest

from src.plassembler.utils.qc import chopper, gzip_compressor_cmd, gzip_file

TEST_DATA = Path("tests/test_data")


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
