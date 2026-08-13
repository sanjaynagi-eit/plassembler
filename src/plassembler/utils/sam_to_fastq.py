import subprocess as sp
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import pysam

# Each read's tally is a small bitmask rather than a set of counters: the
# classification below only ever asks "exactly one alignment?" and "hit a
# plasmid / a chromosome at all?", never for the counts themselves. Every value
# is below 16, so CPython's small-int cache means the dict costs no more than
# its keys - against a dict plus two sets plus a list holding one string per
# alignment before.
HIT_PLASMID = 0b0001
HIT_CHROMOSOME = 0b0010
SEEN = 0b0100
MULTIMAPPED = 0b1000

# flags of a primary alignment, forward and reverse: no secondary (256),
# supplementary (2048) or unmapped (4) bit
PRIMARY_FLAGS = (0, 16)


# fields of a SAM record, 0-based
QNAME, SEQ, QUAL = 0, 9, 10
# SAM's "absent" placeholder, used for RNAME of an unmapped read and for the
# SEQ/QUAL of a record that does not carry them
NO_VALUE = "*"


def _fastq_fields(read):
    """(name, sequence, quality string) straight from the raw SAM record.

    read.query_sequence and read.query_qualities make pysam decode each record
    into python objects - for ONT reads that is a 60,000-element array of ints
    per read, which is then re-encoded to phred+33 one character at a time. The
    SAM line already holds both as strings, so splitting it out is ~11x faster
    and produces byte-identical output.
    """
    fields = read.to_string().split("\t", QUAL + 1)
    return fields[QNAME], fields[SEQ], fields[QUAL]


def _write_record(handle, name, sequence, quality):
    """Write one fastq record."""
    handle.write(f"@{name}\n{sequence}\n+{name}\n{quality}\n")


def _tally_alignments(samname):
    """Per read name: whether it aligned more than once, and whether any of its
    alignments hit a plasmid and/or a chromosome.

    Replaces building a python list holding one string per *alignment* and then
    counting it - for a long-read sam that list is millions of strings.
    """
    tally = defaultdict(int)
    with pysam.AlignmentFile(samname, "r") as samfile:
        for read in samfile.fetch():
            read_name = read.query_name
            previous = tally[read_name]
            flags = previous | SEEN
            if previous & SEEN:
                flags |= MULTIMAPPED
            contig_name = read.reference_name
            if contig_name:
                if "plasmid" in contig_name:
                    flags |= HIT_PLASMID
                elif "chromosome" in contig_name:
                    flags |= HIT_CHROMOSOME
            tally[read_name] = flags
    return tally


def extract_long_fastqs_slow_keep_fastqs(out_dir, samname, plasmidname):
    """Split long reads into plasmid, chromosome and multimapped fastqs.

    Three cheap passes over the sam: one to tally alignments (metadata only, no
    sequences touched), then singly-mapped reads, then multimapped ones. The
    singles-then-multimapped write order is what the previous implementation
    produced and is preserved deliberately, so the downstream assembler sees the
    reads in the same order.
    """
    tally = _tally_alignments(samname)

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
        # singly mapped reads - easy :)
        #################################################
        with pysam.AlignmentFile(samname, "r") as samfile:
            for read in samfile.fetch():
                if tally[read.query_name] & MULTIMAPPED:
                    continue
                contig_name = read.reference_name
                if (contig_name and "plasmid" in contig_name) or read.is_unmapped:
                    target = plasmidfile
                elif contig_name and "chromosome" in contig_name:
                    target = chrom_fastqfile
                else:
                    continue
                _write_record(target, *_fastq_fields(read))

        #################################################
        # multimapped reads - primary alignment only, since the secondary and
        # supplementary records do not carry the full sequence
        #################################################
        with pysam.AlignmentFile(samname, "r") as samfile:
            for read in samfile.fetch():
                flags = tally[read.query_name]
                if not flags & MULTIMAPPED:
                    continue
                if read.flag not in PRIMARY_FLAGS:
                    continue
                hits = flags & (HIT_PLASMID | HIT_CHROMOSOME)
                if hits == (HIT_PLASMID | HIT_CHROMOSOME):
                    target = multimap_plasmid_chromosome_fastqfile
                elif hits == HIT_PLASMID:
                    target = plasmidfile
                elif hits == HIT_CHROMOSOME:
                    target = chrom_fastqfile
                else:
                    continue
                name, sequence, quality = _fastq_fields(read)
                # a record with no qualities carries no usable read
                if quality == NO_VALUE:
                    continue
                _write_record(target, name, sequence, quality)


"""
Thanks to @fanvanf
"""


def extract_long_fastqs_fast(sam_name, plasmidfile, threads):
    cmd = f'samtools  view -@ {threads} {sam_name} | awk \'{{if((($3 ~ /plas/)&& ($2 == "0"|| $2 == "16"))||($2 == "4")) print "@"$1"\\n"$10"\\n+"$1"\\n"$11}}\' > {plasmidfile}'
    # shell=True is required for the samtools | awk pipeline; check=True surfaces
    # a failing samtools instead of silently leaving an empty plasmid FASTQ.
    sp.run(cmd, shell=True, check=True)
