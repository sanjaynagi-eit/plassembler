"""Deterministic (Tier-A) regression tests for pure functions.

These run without the external bioinformatics toolchain and lock the exact
output of pure helper functions against a committed fixture. They are fast and
fully reproducible, so they act as true regression guards.
"""

import shutil
from pathlib import Path

import numpy as np
import pysam
import pytest

from src.plassembler.utils.depth import (
    collate_depths,
    get_contig_circularity,
    get_contig_lengths,
    get_depths_from_bam,
)

GOLDEN_FASTA = Path("tests/test_data/golden/contigs.fasta")


def test_get_contig_lengths_exact():
    """Contig lengths are read exactly from the FASTA."""
    assert get_contig_lengths(GOLDEN_FASTA) == {
        "chromosome": 100,
        "plasmid00001": 60,
        "plasmid00002": 40,
    }


def test_get_contig_circularity_exact():
    """Circularity is 'circular' for the chromosome (by id) and for any contig
    whose description contains 'circular'; everything else is 'not_circular'."""
    assert get_contig_circularity(GOLDEN_FASTA) == {
        "chromosome": "circular",  # id contains "chromosome"
        "plasmid00001": "circular",  # description contains "circular"
        "plasmid00002": "not_circular",
    }


def test_collate_depths_copy_number_math():
    """Copy number is mean plasmid depth divided by mean chromosome depth."""
    depths = {
        "chromosome": [10] * 100,
        "plasmid00001": [20] * 60,  # 2x the chromosome depth
    }
    contig_lengths = {"chromosome": 100, "plasmid00001": 60}

    df = collate_depths(depths, "short", contig_lengths).set_index("contig")

    assert df.loc["chromosome", "mean_depth_short"] == 10
    assert df.loc["plasmid00001", "mean_depth_short"] == 20
    # constant depth -> zero stdev
    assert df.loc["chromosome", "sd_depth_short"] == 0
    # copy numbers relative to the chromosome
    assert df.loc["chromosome", "plasmid_copy_number_short"] == 1.0
    assert df.loc["plasmid00001", "plasmid_copy_number_short"] == 2.0


def test_collate_depths_no_chromosome_is_nan_not_crash():
    """No contig named 'chromosome' -> copy number NaN, not a NameError/crash."""
    depths = {"plasmid00001": [20] * 60}
    contig_lengths = {"plasmid00001": 60}
    df = collate_depths(depths, "short", contig_lengths)
    assert df["plasmid_copy_number_short"].isna().all()
    # must stay float-convertible (downstream does .astype(float))
    df["plasmid_copy_number_short"].astype(float)


def test_collate_depths_zero_chromosome_depth_stays_float():
    """Zero chromosome depth (e.g. the fake --no_chromosome chromosome) -> inf,
    which the depth filter treats as 'keep'. Regression: a string 'NA' here broke
    the downstream .astype(float) in combine_depth_mash_tsvs (CI --no_chromosome)."""
    depths = {"chromosome": [0] * 100, "plasmid00001": [5] * 60}
    contig_lengths = {"chromosome": 100, "plasmid00001": 60}
    df = collate_depths(depths, "short", contig_lengths).set_index("contig")
    assert np.isinf(df.loc["plasmid00001", "plasmid_copy_number_short"])
    # the actual CI failure: this must not raise
    df["plasmid_copy_number_short"].astype(float)


def test_collate_depths_accepts_numpy_arrays():
    """get_depths_from_bam returns numpy arrays; collate_depths must produce the
    same numbers it did for the old python lists (stdev is the *sample* stdev)."""
    from_lists = collate_depths(
        {"chromosome": [10, 12, 8, 10], "plasmid00001": [20, 20, 20, 20]},
        "short",
        {"chromosome": 4, "plasmid00001": 4},
    )
    from_arrays = collate_depths(
        {
            "chromosome": np.array([10, 12, 8, 10], dtype=np.int32),
            "plasmid00001": np.array([20, 20, 20, 20], dtype=np.int32),
        },
        "short",
        {"chromosome": 4, "plasmid00001": 4},
    )
    assert from_lists.to_csv(index=False) == from_arrays.to_csv(index=False)
    # sample stdev of [10, 12, 8, 10] is 1.63, the population one would be 1.41
    assert from_arrays.set_index("contig").loc["chromosome", "sd_depth_short"] == 1.63


def test_collate_depths_short_contig_is_na():
    """Fewer than 2 bases has no sample stdev, so all four columns are NA - the
    old code got there via statistics.StatisticsError."""
    df = collate_depths(
        {"plasmid00001": np.array([7], dtype=np.int32)}, "long", {"plasmid00001": 1}
    ).set_index("contig")
    assert df.loc["plasmid00001", "mean_depth_long"] == "NA"
    assert df.loc["plasmid00001", "sd_depth_long"] == "NA"
    assert df.loc["plasmid00001", "q25_depth_long"] == "NA"
    assert df.loc["plasmid00001", "q75_depth_long"] == "NA"


@pytest.mark.requires_tool
@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools not installed")
def test_get_depths_from_bam_exact(tmp_path):
    """Per-base depths are scattered into zero-filled arrays: positions samtools
    never reports stay 0, and contigs with no alignments at all stay all-zero."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [
            {"SN": "chromosome", "LN": 10},
            {"SN": "plasmid00001", "LN": 6},
            {"SN": "plasmid00002", "LN": 4},  # deliberately gets no reads
        ],
    }
    bam_path = tmp_path / "test.bam"
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for ref_id, start, length in [(0, 2, 4), (0, 4, 4), (1, 0, 3)]:
            read = pysam.AlignedSegment()
            read.query_name = f"r{ref_id}_{start}"
            read.query_sequence = "A" * length
            read.query_qualities = pysam.qualitystring_to_array("I" * length)
            read.flag = 0
            read.reference_id = ref_id
            read.reference_start = start
            read.mapping_quality = 60
            read.cigarstring = f"{length}M"
            bam.write(read)
    pysam.index(str(bam_path))

    depths = get_depths_from_bam(
        bam_path, {"chromosome": 10, "plasmid00001": 6, "plasmid00002": 4}
    )
    # reads cover chromosome 3-6 and 5-8 (1-based), so 5-6 are doubly covered
    np.testing.assert_array_equal(
        depths["chromosome"], np.array([0, 0, 1, 1, 2, 2, 1, 1, 0, 0])
    )
    np.testing.assert_array_equal(depths["plasmid00001"], np.array([1, 1, 1, 0, 0, 0]))
    np.testing.assert_array_equal(depths["plasmid00002"], np.zeros(4, dtype=np.int32))


@pytest.mark.requires_tool
@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools not installed")
def test_get_depths_from_bam_empty(tmp_path):
    """An alignment-free bam yields all-zero arrays rather than raising on the
    empty `samtools depth` stream."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chromosome", "LN": 5}]}
    bam_path = tmp_path / "empty.bam"
    with pysam.AlignmentFile(bam_path, "wb", header=header):
        pass
    depths = get_depths_from_bam(bam_path, {"chromosome": 5})
    np.testing.assert_array_equal(depths["chromosome"], np.zeros(5, dtype=np.int32))


@pytest.mark.requires_tool
@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools not installed")
def test_get_depths_from_bam_rejects_an_unknown_contig(tmp_path):
    """A contig in the bam but not in the reference means the two disagree. The
    old dict-indexing raised KeyError here; keep failing rather than silently
    dropping that contig's coverage."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "unexpected_contig", "LN": 8}]}
    bam_path = tmp_path / "mismatch.bam"
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        read = pysam.AlignedSegment()
        read.query_name = "r1"
        read.query_sequence = "ACGT"
        read.query_qualities = pysam.qualitystring_to_array("IIII")
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 0
        read.mapping_quality = 60
        read.cigarstring = "4M"
        bam.write(read)

    with pytest.raises(KeyError, match="unexpected_contig"):
        get_depths_from_bam(bam_path, {"chromosome": 100})
