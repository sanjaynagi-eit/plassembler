"""Tests for mash tophit selection in plassembler.utils.run_mash."""

import pandas as pd
import pytest

from src.plassembler.utils.run_mash import TOPHITS_COLUMNS, mash_tophits_df

COL_LIST = [
    "contig",
    "NUCCORE_ACC",
    "mash_distance",
    "mash_pval",
    "mash_matching_hashes",
]


def write_mash(path, rows):
    pd.DataFrame(rows, columns=COL_LIST).to_csv(
        path, sep="\t", header=False, index=False
    )
    return path


def test_tophits_picks_lowest_distance_per_contig(tmp_path):
    tsv = write_mash(
        tmp_path / "mash.tsv",
        [
            [1, "ACC_far", 0.09, 0.0, "10/1000"],
            [1, "ACC_near", 0.01, 0.0, "900/1000"],
            [2, "ACC_two", 0.05, 0.0, "500/1000"],
        ],
    )
    df = mash_tophits_df(tsv, 2).set_index("contig")
    assert df.loc[1, "NUCCORE_ACC"] == "ACC_near"
    assert df.loc[1, "mash_distance"] == 0.01
    assert df.loc[2, "NUCCORE_ACC"] == "ACC_two"
    assert (df["PLSDB_hit"] == "Yes").all()


def test_tophits_has_a_row_per_contig_including_misses(tmp_path):
    """Contigs with no mash hit still get a row, with blanks - downstream code
    indexes by contig and would silently drop them otherwise."""
    tsv = write_mash(tmp_path / "mash.tsv", [[2, "ACC", 0.05, 0.0, "500/1000"]])
    df = mash_tophits_df(tsv, 3)
    assert list(df["contig"]) == [1, 2, 3]
    assert list(df.columns) == TOPHITS_COLUMNS
    assert df.set_index("contig").loc[1, "PLSDB_hit"] == ""
    assert df.set_index("contig").loc[1, "NUCCORE_ACC"] == ""
    assert df.set_index("contig").loc[3, "mash_distance"] == ""


def test_tophits_empty_mash_file(tmp_path):
    """No hits at all: still one blank row per contig."""
    tsv = tmp_path / "mash.tsv"
    tsv.write_text("")
    df = mash_tophits_df(tsv, 2)
    assert list(df["contig"]) == [1, 2]
    assert (df["PLSDB_hit"] == "").all()
    assert list(df.columns) == TOPHITS_COLUMNS


def test_tophits_tie_broken_by_mash_output_order(tmp_path):
    """Equal distances resolve to the first row in mash's output, deterministically.
    The old per-contig unstable sort left this arbitrary."""
    tsv = write_mash(
        tmp_path / "mash.tsv",
        [
            [1, "ACC_first", 0.02, 0.0, "500/1000"],
            [1, "ACC_second", 0.02, 0.0, "500/1000"],
            [1, "ACC_third", 0.02, 0.0, "500/1000"],
        ],
    )
    assert mash_tophits_df(tsv, 1).loc[0, "NUCCORE_ACC"] == "ACC_first"


@pytest.mark.parametrize("contig_count", [1, 5])
def test_tophits_row_count_matches_contig_count(tmp_path, contig_count):
    """Extra contigs in the mash output must not add rows beyond contig_count."""
    tsv = write_mash(
        tmp_path / "mash.tsv",
        [[c, f"ACC{c}", 0.01 * c, 0.0, "1/1000"] for c in range(1, 9)],
    )
    assert len(mash_tophits_df(tsv, contig_count)) == contig_count
