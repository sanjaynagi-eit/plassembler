import subprocess as sp
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import pysam

from plassembler.utils.external_tools import ExternalTool
from plassembler.utils.mapping import minimap2_model_for


def extract_long_fastqs_slow_keep_fastqs(out_dir, samname, plasmidname):
    #################################################
    # Get the single and multiple map reads as sets
    #################################################

    # get list of all read names
    read_names = []
    with pysam.AlignmentFile(samname, "r") as samfile:
        for read in samfile.fetch():
            read_names.append(read.query_name)

    # count occurrences of each read name
    count_dict = defaultdict(int)
    for item in read_names:
        count_dict[item] += 1

    # sets give O(1) membership (the per-read lookups below run once per
    # alignment, so lists here would make the whole function quadratic)
    single_read_names = {name for name, count in count_dict.items() if count == 1}
    multi_read_names = {name for name, count in count_dict.items() if count != 1}

    # ExitStack guarantees all output handles are closed even if a write or a
    # pysam call raises partway through
    with ExitStack() as stack:
        # reads mapping to plasmids, or not mapping to any contigs
        plasmidfile = stack.enter_context(open(plasmidname, "w"))
        # reads mapping to multiple contigs
        multimap_plasmid_chromosome_fastqfile = stack.enter_context(
            open(Path(out_dir) / "multimap_plasmid_chromosome_long.fastq", "w")
        )
        # chromosome fastqs
        chrom_fastqfile = stack.enter_context(
            open(Path(out_dir) / "chromosome_mapped_long.fastq", "w")
        )

        #################################################
        # process all single reads and count plasmid vs chromosome multimaps
        #################################################

        plasmid_mm_dict = defaultdict(int)
        chromosome_mm_dict = defaultdict(int)

        with pysam.AlignmentFile(samname, "r") as samfile:
            for read in samfile.fetch():
                read_name = read.query_name
                sequence = read.query_sequence
                quality = read.query_qualities
                # get contig name for the read
                contig_name = samfile.get_reference_name(read.reference_id)

                # single reads - easy :)
                if read_name in single_read_names:
                    # plasmid-mapped reads and all unmapped reads
                    if (contig_name and "plasmid" in contig_name) or read.is_unmapped:
                        plasmidfile.write(f"@{read_name}\n")
                        plasmidfile.write(f"{sequence}\n")
                        plasmidfile.write(f"+{read_name}\n")
                        plasmidfile.write("".join(chr(q + 33) for q in quality) + "\n")
                    elif contig_name and "chromosome" in contig_name:
                        chrom_fastqfile.write(f"@{read_name}\n")
                        chrom_fastqfile.write(f"{sequence}\n")
                        chrom_fastqfile.write(f"+{read_name}\n")
                        chrom_fastqfile.write(
                            "".join(chr(q + 33) for q in quality) + "\n"
                        )
                # build count dictionaries for the multimap reads (next step)
                else:
                    if contig_name and "plasmid" in contig_name:
                        plasmid_mm_dict[read_name] += 1
                    elif contig_name and "chromosome" in contig_name:
                        chromosome_mm_dict[read_name] += 1

        #################################################
        # process all multimap reads
        #################################################

        with pysam.AlignmentFile(samname, "r") as samfile:
            for read in samfile.fetch():
                read_name = read.query_name
                sequence = read.query_sequence
                quality = read.query_qualities
                flag = read.flag

                if read_name in multi_read_names:
                    # multimap to both plasmid and chromosome
                    if (
                        plasmid_mm_dict[read_name] > 0
                        and chromosome_mm_dict[read_name] > 0
                    ):
                        if quality is not None and (flag == 0 or flag == 16):
                            # get only the primary
                            multimap_plasmid_chromosome_fastqfile.write(
                                f"@{read_name}\n"
                            )
                            multimap_plasmid_chromosome_fastqfile.write(f"{sequence}\n")
                            multimap_plasmid_chromosome_fastqfile.write(
                                f"+{read_name}\n"
                            )
                            multimap_plasmid_chromosome_fastqfile.write(
                                "".join(chr(q + 33) for q in quality) + "\n"
                            )
                    # multimap to plasmid only -> plasmid file
                    elif plasmid_mm_dict[read_name] > 0:
                        if quality is not None and (flag == 0 or flag == 16):
                            plasmidfile.write(f"@{read_name}\n")
                            plasmidfile.write(f"{sequence}\n")
                            plasmidfile.write(f"+{read_name}\n")
                            plasmidfile.write(
                                "".join(chr(q + 33) for q in quality) + "\n"
                            )
                    # multimap to chromosome only -> chromosome file
                    elif chromosome_mm_dict[read_name] > 0:
                        if quality is not None and (flag == 0 or flag == 16):
                            chrom_fastqfile.write(f"@{read_name}\n")
                            chrom_fastqfile.write(f"{sequence}\n")
                            chrom_fastqfile.write(f"+{read_name}\n")
                            chrom_fastqfile.write(
                                "".join(chr(q + 33) for q in quality) + "\n"
                            )


"""
Thanks to @fanvanf
"""


# keep reads whose primary alignment (flag 0 or 16) is to a contig whose name
# contains "plas", plus every unmapped read (flag 4), and emit them as fastq
PLASMID_READ_AWK = (
    '{if((($3 ~ /plas/)&& ($2 == "0"|| $2 == "16"))||($2 == "4"))'
    ' print "@"$1"\\n"$10"\\n+"$1"\\n"$11}'
)


def extract_long_fastqs_fast(sam_name, plasmidfile, threads):
    cmd = f"samtools  view -@ {threads} {sam_name} | awk '{PLASMID_READ_AWK}' > {plasmidfile}"
    # shell=True is required for the samtools | awk pipeline; check=True surfaces
    # a failing samtools instead of silently leaving an empty plasmid FASTQ.
    sp.run(cmd, shell=True, check=True)


def map_and_extract_long_fastqs(
    input_long_reads, fasta, plasmidfile, threads, pacbio_model, logdir
):
    """Map long reads and pull the plasmid/unmapped ones straight out of the stream.

    In long-only mode `long_read.sam` has exactly one consumer - this extraction
    - so writing minimap2's full uncompressed SAM to disk only to have samtools
    read it straight back is pure I/O. Piping the three stages together removes
    it entirely: on a real ONT isolate that is ~0.6 GiB written and re-read.

    :param input_long_reads: reads to map
    :param fasta: reference (renamed flye assembly)
    :param plasmidfile: output fastq of plasmid and unmapped reads
    :param threads: threads
    :param pacbio_model: pacbio_model
    :param logdir: logdir
    :return:
    """
    minimap2_model = minimap2_model_for(pacbio_model)
    minimap2 = ExternalTool(
        tool="minimap2",
        input="",
        output="",
        params=f" -ax {minimap2_model} -t {threads} {fasta} {input_long_reads}",
        logdir=logdir,
        outfile="",
    )
    samtools_view = ExternalTool(
        tool="samtools",
        input="",
        output="",
        params=f" view -@ {threads}",
        logdir=logdir,
        outfile="",
    )
    awk = ExternalTool(
        tool="awk",
        input="",
        output="",
        params=f" '{PLASMID_READ_AWK}'",
        logdir=logdir,
        outfile="",
    )
    ExternalTool.run_piped((minimap2, samtools_view, awk), outfile=plasmidfile)
