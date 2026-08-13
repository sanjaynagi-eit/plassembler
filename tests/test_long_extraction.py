"""Tests for mapping long reads straight into the plasmid FASTQ."""

import hashlib
import shutil
from pathlib import Path

import pytest

from src.plassembler.utils.mapping import minimap_long_reads
from src.plassembler.utils.sam_to_fastq import (
    PLASMID_READ_AWK,
    extract_long_fastqs_fast,
    map_and_extract_long_fastqs,
)

TEST_DATA = Path("tests/test_data")
TOOLS_PRESENT = all(shutil.which(tool) for tool in ("minimap2", "samtools", "awk"))


def test_awk_program_is_the_documented_filter():
    """Primary plasmid alignments (flag 0/16) plus every unmapped read (flag 4)."""
    assert "$3 ~ /plas/" in PLASMID_READ_AWK
    assert '$2 == "0"' in PLASMID_READ_AWK
    assert '$2 == "16"' in PLASMID_READ_AWK
    assert '$2 == "4"' in PLASMID_READ_AWK


@pytest.fixture
def reference(tmp_path):
    """A chromosome and a plasmid, named the way identify_chromosome_process_*
    names them - the awk filter keys off "plas" in the contig name."""
    fasta = tmp_path / "flye_renamed.fasta"
    fasta.write_text(
        ">chromosome\n" + "ACGTTGCA" * 400 + "\n>plasmid_1\n" + "GGCCTTAA" * 200 + "\n"
    )
    return fasta


@pytest.mark.requires_tool
@pytest.mark.skipif(not TOOLS_PRESENT, reason="minimap2/samtools/awk not installed")
def test_piped_extraction_matches_the_two_step_form(tmp_path, reference):
    """The piped pipeline must produce exactly the FASTQ that writing a SAM and
    running samtools|awk over it produced."""
    reads = str(TEST_DATA / "test_long.fastq.gz")
    logdir = tmp_path / "logs"

    two_step = tmp_path / "two_step.fastq"
    sam = tmp_path / "long_read.sam"
    minimap_long_reads(reads, reference, sam, "2", "nothing", logdir)
    extract_long_fastqs_fast(sam, two_step, "2")

    piped = tmp_path / "piped.fastq"
    map_and_extract_long_fastqs(reads, reference, piped, "2", "nothing", logdir)

    assert (
        hashlib.sha256(piped.read_bytes()).hexdigest()
        == hashlib.sha256(two_step.read_bytes()).hexdigest()
    )


@pytest.mark.requires_tool
@pytest.mark.skipif(not TOOLS_PRESENT, reason="minimap2/samtools/awk not installed")
def test_piped_extraction_writes_no_intermediate_sam(tmp_path, reference):
    """The point of the change: nothing lands on disk between the stages."""
    reads = str(TEST_DATA / "test_long.fastq.gz")
    out = tmp_path / "plasmid_long.fastq"
    map_and_extract_long_fastqs(
        reads, reference, out, "2", "nothing", tmp_path / "logs"
    )

    assert out.exists()
    assert list(tmp_path.glob("*.sam")) == []


@pytest.mark.requires_tool
@pytest.mark.skipif(not TOOLS_PRESENT, reason="minimap2/samtools/awk not installed")
def test_piped_extraction_output_is_valid_fastq(tmp_path, reference):
    reads = str(TEST_DATA / "test_long.fastq.gz")
    out = tmp_path / "plasmid_long.fastq"
    map_and_extract_long_fastqs(
        reads, reference, out, "2", "nothing", tmp_path / "logs"
    )

    lines = out.read_text().splitlines()
    assert len(lines) % 4 == 0
    for i in range(0, len(lines), 4):
        assert lines[i].startswith("@")
        assert lines[i + 2].startswith("+")
        assert len(lines[i + 1]) == len(lines[i + 3])
