from pathlib import Path

from plassembler.utils.external_tools import ExternalTool

#################################
# original mapping
#################################


def minimap2_model_for(pacbio_model):
    """maps plassembler's --pacbio_model onto a minimap2 preset
    :param pacbio_model: pacbio_model
    :return: minimap2 preset name
    """
    if pacbio_model in ("--pacbio-raw", "--pacbio-corr"):
        return "map-pb"
    if pacbio_model == "--pacbio-hifi":
        return "map-hifi"
    # ONT, and the "nothing" default
    return "map-ont"


def _minimap2_tool(params, logdir, outfile=""):
    return ExternalTool(
        tool="minimap2",
        input="",
        output="",
        params=params,
        logdir=logdir,
        outfile=outfile,
    )


def _samtools_sort_tool(sorted_bam, threads, logdir):
    return ExternalTool(
        tool="samtools",
        input="",
        output="",
        params=f" sort -@ {threads} -o {sorted_bam}",
        logdir=logdir,
        outfile="",
    )


def minimap_long_reads_to_sorted_bam(
    input_long_reads, fasta, sorted_bam: Path, threads, pacbio_model, logdir
):
    """maps long reads with minimap2 straight into a sorted bam

    minimap2's SAM was previously written to disk in full and then re-read by
    samtools sort. For an ONT isolate that is ~0.6 GiB written and read back per
    mapping, twice per run, purely as a pipe buffer.

    :param input_long_reads: reads to map
    :param fasta: reference
    :param sorted_bam: output sorted bam
    :param threads: threads
    :param pacbio_model: pacbio_model
    :param logdir: logdir
    :return:
    """
    minimap2_model = minimap2_model_for(pacbio_model)
    ExternalTool.run_piped(
        (
            _minimap2_tool(
                f" -ax {minimap2_model} -t {threads} {fasta} {input_long_reads}", logdir
            ),
            _samtools_sort_tool(sorted_bam, threads, logdir),
        )
    )


def minimap_short_reads_to_sorted_bam(r1, r2, fasta, sorted_bam: Path, threads, logdir):
    """maps short reads with minimap2 straight into a sorted bam
    :param r1: R1 reads
    :param r2: R2 reads
    :param fasta: reference
    :param sorted_bam: output sorted bam
    :param threads: threads
    :param logdir: logdir
    :return:
    """
    ExternalTool.run_piped(
        (
            _minimap2_tool(f" -ax sr -t {threads} {fasta} {r1} {r2}", logdir),
            _samtools_sort_tool(sorted_bam, threads, logdir),
        )
    )


def minimap_long_reads(input_long_reads, fasta, sam, threads, pacbio_model, logdir):
    """maps long reads using minimap2
    :param threads: threads
    :param pacbio_model: pacbio_model
    :param threads: threads
    :param logdir: logdir
    :return:
    """

    minimap2_model = minimap2_model_for(pacbio_model)

    minimap2 = _minimap2_tool(
        f" -ax {minimap2_model} -t {threads} {fasta} {input_long_reads}",
        logdir,
        outfile=sam,
    )

    # need to write to stdout
    ExternalTool.run_tool(minimap2, to_stdout=True)


# short reads


def minimap_short_reads(r1, r2, fasta, sam, threads, logdir):
    """maps short reads using minimap2
    :param outdir: output directory path
    :param threads: threads
    :param logdir: logdir
    :return:
    """

    minimap2 = ExternalTool(
        tool="minimap2",
        input="",
        output="",
        params=f" -ax sr -t {threads} {fasta} {r1} {r2}",
        logdir=logdir,
        outfile=sam,
    )

    # need to write to stdout
    ExternalTool.run_tool(minimap2, to_stdout=True)
