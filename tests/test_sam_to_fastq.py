"""Tests for splitting a long-read SAM into plasmid/chromosome/multimap FASTQs."""

import pysam
import pytest

from src.plassembler.utils.sam_to_fastq import extract_long_fastqs_slow_keep_fastqs

HEADER = {
    "HD": {"VN": "1.6"},
    "SQ": [{"SN": "chromosome", "LN": 200}, {"SN": "plasmid_1", "LN": 100}],
}


def aligned(name, ref_id, start, seq, flag=0):
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = seq
    read.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    read.flag = flag
    read.reference_id = ref_id
    read.reference_start = start
    read.mapping_quality = 60
    read.cigarstring = f"{len(seq)}M"
    return read


def unaligned(name, seq):
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = seq
    read.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    read.flag = 4
    read.reference_id = -1
    read.reference_start = -1
    return read


def write_sam(path, reads):
    with pysam.AlignmentFile(path, "w", header=HEADER) as sam:
        for read in reads:
            sam.write(read)
    return path


def names_in(path):
    return [
        line[1:].strip()
        for i, line in enumerate(open(path).read().splitlines())
        if i % 4 == 0
    ]


@pytest.fixture
def outputs(tmp_path):
    return (
        tmp_path / "plasmid_long.fastq",
        tmp_path / "multimap_plasmid_chromosome_long.fastq",
        tmp_path / "chromosome_mapped_long.fastq",
    )


def test_singly_mapped_reads_are_routed_by_contig(tmp_path, outputs):
    plasmid, multimap, chrom = outputs
    sam = write_sam(
        tmp_path / "long_read.sam",
        [
            aligned("chrom_read", 0, 10, "ACGT" * 5),
            aligned("plasmid_read", 1, 10, "TTTT" * 5),
            unaligned("unmapped_read", "GGGG" * 5),
        ],
    )
    extract_long_fastqs_slow_keep_fastqs(tmp_path, sam, plasmid)

    # unmapped reads go to the plasmid file: they may come from a plasmid the
    # assembly missed entirely
    assert sorted(names_in(plasmid)) == ["plasmid_read", "unmapped_read"]
    assert names_in(chrom) == ["chrom_read"]
    assert names_in(multimap) == []


def test_read_hitting_both_goes_to_the_multimap_file(tmp_path, outputs):
    plasmid, multimap, chrom = outputs
    sam = write_sam(
        tmp_path / "long_read.sam",
        [
            aligned("both", 0, 10, "ACGT" * 5),
            aligned("both", 1, 10, "ACGT" * 5, flag=256),  # secondary
        ],
    )
    extract_long_fastqs_slow_keep_fastqs(tmp_path, sam, plasmid)

    # only the primary record is written, once
    assert names_in(multimap) == ["both"]
    assert names_in(plasmid) == []
    assert names_in(chrom) == []


def test_read_multimapping_within_one_replicon(tmp_path, outputs):
    """Two plasmid alignments is still a plasmid read, written once."""
    plasmid, multimap, chrom = outputs
    sam = write_sam(
        tmp_path / "long_read.sam",
        [
            aligned("plas_twice", 1, 5, "ACGT" * 5),
            aligned("plas_twice", 1, 40, "ACGT" * 5, flag=256),
        ],
    )
    extract_long_fastqs_slow_keep_fastqs(tmp_path, sam, plasmid)
    assert names_in(plasmid) == ["plas_twice"]
    assert names_in(multimap) == []


def test_sequence_and_quality_survive_the_round_trip(tmp_path, outputs):
    """Fields come from the raw SAM record rather than pysam's decoded objects,
    so pin that a read's bases and qualities are written unchanged."""
    plasmid, _, chrom = outputs
    seq = "ACGTACGTTT"
    read = aligned("chrom_read", 0, 10, seq)
    read.query_qualities = pysam.qualitystring_to_array("I!I!I!I!I!")
    sam = write_sam(tmp_path / "long_read.sam", [read])
    extract_long_fastqs_slow_keep_fastqs(tmp_path, sam, plasmid)

    lines = chrom.read_text().splitlines()
    assert lines == ["@chrom_read", seq, "+chrom_read", "I!I!I!I!I!"]


def test_reverse_strand_read_is_written_as_stored(tmp_path, outputs):
    """A flag-16 alignment stores the reverse complement in SEQ; the old code
    took the same field via query_sequence, so output must not change."""
    plasmid, _, chrom = outputs
    sam = write_sam(
        tmp_path / "long_read.sam", [aligned("rev", 0, 10, "AAAACCCCGG", flag=16)]
    )
    extract_long_fastqs_slow_keep_fastqs(tmp_path, sam, plasmid)
    assert chrom.read_text().splitlines()[1] == "AAAACCCCGG"


def test_empty_sam_produces_empty_outputs(tmp_path, outputs):
    plasmid, multimap, chrom = outputs
    sam = write_sam(tmp_path / "long_read.sam", [])
    extract_long_fastqs_slow_keep_fastqs(tmp_path, sam, plasmid)
    for path in (plasmid, multimap, chrom):
        assert path.exists() and path.read_text() == ""
