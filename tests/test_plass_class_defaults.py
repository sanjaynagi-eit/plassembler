"""Regression tests for constructor defaults and the chromid warning."""

import pandas as pd

from src.plassembler.utils.plass_class import Assembly, Plass


def test_plass_mutable_defaults_are_not_shared():
    """A mutable default is created once at import and shared by every instance."""
    first, second = Plass(), Plass()
    assert first.filtered_out_contig_ids is not second.filtered_out_contig_ids
    first.filtered_out_contig_ids.append("1")
    assert second.filtered_out_contig_ids == []


def test_assembly_mutable_defaults_are_not_shared():
    first, second = Assembly(), Assembly()
    assert first.plasmid_names is not second.plasmid_names
    first.plasmid_names.append("2")
    assert second.plasmid_names == ["1"]


def test_dataframe_defaults_are_empty_and_per_instance():
    """The defaults used to be a dummy {'col1': [1,2,3]} frame built at import."""
    first, second = Plass(), Plass()
    for frame in (first.depth_df, first.mash_df, first.combined_depth_mash_df):
        assert isinstance(frame, pd.DataFrame)
        assert frame.empty
    assert first.depth_df is not second.depth_df


def test_supplied_values_are_still_used():
    df = pd.DataFrame({"contig": ["1"]})
    plass = Plass(depth_df=df, filtered_out_contig_ids=["7"])
    assert plass.depth_df is df
    assert plass.filtered_out_contig_ids == ["7"]


def test_multiple_chromosome_warning_can_fire(tmp_path, caplog):
    """`c` is an int, so the old `if c == "2"` could never be true and the
    "multiple contigs above -c" warning was unreachable."""
    from loguru import logger

    assembly = tmp_path / "assembly.fasta"
    # two contigs above the 1000 bp chromosome threshold, plus one below
    assembly.write_text(
        ">contig_1\n" + "A" * 2000 + "\n"
        ">contig_2\n" + "C" * 1500 + "\n"
        ">contig_3\n" + "G" * 100 + "\n"
    )

    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        plass = Plass()
        plass.outdir = str(tmp_path)
        plass.identify_chromosome_process_raven(1000)
    finally:
        logger.remove(sink_id)

    assert any("Multiple contigs above the specified chromosome length" in m for m in messages)
    assert plass.chromosome_flag is True
