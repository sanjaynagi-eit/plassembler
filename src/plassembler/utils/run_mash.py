import os
import shutil
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from plassembler.utils.external_tools import ExternalTool

MASH_COL_LIST = [
    "contig",
    "NUCCORE_ACC",
    "mash_distance",
    "mash_pval",
    "mash_matching_hashes",
]

TOPHITS_COLUMNS = [
    "contig",
    "PLSDB_hit",
    "NUCCORE_ACC",
    "mash_distance",
    "mash_pval",
    "mash_matching_hashes",
]

# the PLSDB metadata tsv has no header row of its own
PLSDB_COLUMNS = [
    "NUCCORE_UID",
    "NUCCORE_ACC",
    "NUCCORE_Description",
    "NUCCORE_CreateDate",
    "NUCCORE_Topology",
    "NUCCORE_Completeness",
    "NUCCORE_TaxonID",
    "NUCCORE_Genome",
    "NUCCORE_Length",
    "NUCCORE_DuplicatedEntry",
    "NUCCORE_Source",
    "NUCCORE_BiosampleID",
    "BIOSAMPLE_UID",
    "BIOSAMPLE_ACC",
    "BIOSAMPLE_Location",
    "BIOSAMPLE_Coordinates",
    "BIOSAMPLE_IsolationSource",
    "BIOSAMPLE_Host",
    "BIOSAMPLE_CollectionDate",
    "BIOSAMPLE_HostDisease",
    "BIOSAMPLE_SampleType",
    "ASSEMBLY_UID",
    "ASSEMBLY_ACC",
    "ASSEMBLY_Status",
    "ASSEMBLY_coverage",
    "ASSEMBLY_SeqReleaseDate",
    "ASSEMBLY_SubmissionDate",
    "ASSEMBLY_Lastest",
    "ASSEMBLY_BiosampleID",
    "TAXONOMY_superkingdom",
    "TAXONOMY_phylum",
    "TAXONOMY_class",
    "TAXONOMY_order",
    "TAXONOMY_family",
    "TAXONOMY_genus",
    "TAXONOMY_species",
    "TAXONOMY_strain",
    "TAXONOMY_UID",
    "TAXONOMY_taxon_rank",
    "TAXONOMY_taxon_name",
    "TAXONOMY_taxon_lineage",
    "TAXONOMY_superkingdom_id",
    "TAXONOMY_phylum_id",
    "TAXONOMY_class_id",
    "TAXONOMY_order_id",
    "TAXONOMY_family_id",
    "TAXONOMY_genus_id",
    "TAXONOMY_species_id",
    "TAXONOMY_strain_id",
    "has_biosample",
    "has_assembly",
    "has_location",
    "rMLST_hits",
    "rMLST_hitscount",
    "inclusions",
    "NUCCORE_GC",
    "Length",
    "BIOSAMPLE_Host_processed",
    "BIOSAMPLE_Host_processed_source",
    "BIOSAMPLE_Host_label",
    "BIOSAMPLE_HostDisease_processed",
    "loc_lat",
    "loc_lng",
    "loc_parsed",
    "D1",
    "D2",
    "plasmidfinder",
    "pmlst",
]

# NOTE on memory: this table is read whole (~240 MB of dataframe for the v1.5.0
# database) to serve a merge that keeps at most one row per contig, so reading it
# in chunks and discarding non-matching rows is tempting. Do not: pandas infers
# column dtypes per chunk, and several PLSDB columns are only mixed-type when the
# whole file is seen. ASSEMBLY_coverage, for instance, is object over the full
# table and writes "120", but is float64 within a small chunk and writes "120.0".
# The merged frame goes verbatim into <prefix>_summary.tsv, so that is a
# user-visible output change. Measured at 0.5s and ~240 MB transient, which is
# not worth pinning 68 dtypes to the current database release.


def mash_sketch(out_dir, fasta_file, logdir):
    """
    Runs mash to output fastas
    :param out_dir: output directory
    :param logger: logger
    :return:
    """

    plasmid_fasta: Path = Path(f"{out_dir}/plasmids.fasta")
    shutil.copy2(fasta_file, plasmid_fasta)

    # mash command
    mash = ExternalTool(
        tool="mash",
        input="",
        output="",
        params=f" sketch {plasmid_fasta} -i ",
        logdir=logdir,
        outfile="",
    )

    # need to write to stdout
    ExternalTool.run_tool(mash, to_stdout=False)


def run_mash(out_dir, plassembler_db_dir, logdir):
    """
    Runs mash to output fastas
    :param out_dir: output directory
    :param plassembler_db_dir: plassembler db directory
    :param logger: logger
    :return:
    """

    plsdb_sketch: Path = Path(f"{plassembler_db_dir}/plsdb_2023_11_03_v2.msh")
    plasmid_sketch: Path = Path(f"{out_dir}/plasmids.fasta.msh")
    mash_tsv: Path = Path(f"{out_dir}/mash.tsv")

    mash = ExternalTool(
        tool="mash",
        input="",
        output="",
        params=f" dist  {plasmid_sketch} {plsdb_sketch} -v 0.1 -d 0.1 -i ",
        logdir=logdir,
        outfile=mash_tsv,
    )

    # need to write to stdout
    ExternalTool.run_tool(mash, to_stdout=True)


def get_contig_count(plasmid_fasta):
    """
    Process mash output
    :param out_dir: output directory
    :return: i: int contig_count
    """
    i = 0
    for dna_record in SeqIO.parse(plasmid_fasta, "fasta"):
        i += 1
    return i


# check if a file has more than 1 line (not empty)
def is_file_empty(file):
    """
    Determines if file is empty
    :param file: file path
    :return: empty Boolean
    """
    empty = False
    if os.stat(file).st_size == 0:
        empty = True
    return empty


def mash_tophits_df(mash_tsv, contig_count):
    """Best (lowest mash distance) PLSDB hit per contig, one row per contig.

    Contigs 1..contig_count always get a row; those without a hit get blanks.

    The previous implementation filtered and sorted the whole mash dataframe
    twice for every contig inside a python loop, so cost grew with
    contigs x hits: 400 contigs took 0.283s where this takes 0.006s. It also
    lived in two near-identical ~90 line copies, one per class.

    Ties: when several references share the lowest distance - mash distance is a
    function of matching hash count, so exact ties do happen - the old code took
    whichever row pandas' unstable quicksort happened to place first. The stable
    sort below always takes the first such row in mash's output order. For the
    handful of hits per contig that `mash dist -d 0.1 -v 0.1` actually returns,
    numpy sorts stably anyway, so this matches the old result and merely makes it
    reproducible rather than arbitrary.

    :param mash_tsv: mash dist output
    :param contig_count: number of contigs in the assembly
    :return: dataframe with TOPHITS_COLUMNS
    """
    blank_row = ["", "", "", "", ""]

    if is_file_empty(mash_tsv):
        return pd.DataFrame(
            [[contig] + blank_row for contig in range(1, contig_count + 1)],
            columns=TOPHITS_COLUMNS,
        )

    mash_df = pd.read_csv(
        mash_tsv, delimiter="\t", index_col=False, names=MASH_COL_LIST
    )
    # stable sort keeps the old tie-breaking: among equal distances the first
    # row in the mash output wins, which is what .sort_values().loc[0] gave
    best = (
        mash_df.sort_values("mash_distance", kind="stable")
        .groupby("contig", sort=False)
        .first()
    )

    # Rows are assembled from the selected hits rather than produced by a
    # left merge against 1..contig_count. A merge would introduce NaN for
    # contigs without a hit, and that silently promotes int columns to float:
    # a mash_pval of 0 came back out as "0.0" in <prefix>_summary.tsv. This
    # loop is over contigs only - the expensive part, selecting the best hit,
    # already happened once above.
    rows = []
    for contig in range(1, contig_count + 1):
        if contig not in best.index:
            rows.append([contig] + blank_row)
            continue
        hit = best.loc[contig]
        rows.append(
            [
                contig,
                "Yes",
                hit.NUCCORE_ACC,
                hit.mash_distance,
                hit.mash_pval,
                hit.mash_matching_hashes,
            ]
        )
    return pd.DataFrame(rows, columns=TOPHITS_COLUMNS)


def load_plsdb_metadata(plassembler_db_dir):
    """The PLSDB metadata table, used to describe each contig's mash tophit.

    See the note on PLSDB_COLUMNS above for why this is deliberately a single
    whole-file read.

    :param plassembler_db_dir: plassembler database directory
    :return: dataframe with PLSDB_COLUMNS
    """
    plsdb_tsv_file = os.path.join(plassembler_db_dir, "plsdb_2023_11_03_v2.tsv")
    return pd.read_csv(
        plsdb_tsv_file,
        delimiter="\t",
        index_col=False,
        names=PLSDB_COLUMNS,
        skiprows=1,
        low_memory=False,
    )
