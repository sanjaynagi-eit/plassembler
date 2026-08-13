import gzip
import shutil
import subprocess as sp
from contextlib import ExitStack
from pathlib import Path

from loguru import logger

from plassembler.utils.external_tools import ExternalTool


def gzip_compressor_cmd(threads):
    """Command that reads plain data on stdin and writes a gzip stream to stdout.

    Prefers bgzip, which ships with htslib/samtools (already a hard plassembler
    dependency) and compresses in parallel. BGZF is a valid gzip stream, so every
    downstream reader - flye, minimap2, chopper, gunzip, python's gzip module -
    handles the result unchanged.

    Serial gzip dominated the chopper stage: on a 300 MB ONT fastq the chain took
    47.7s, of which only 5.4s was gunzip+chopper and 42s was gzip.

    :param threads: thread count (str or int) to hand to bgzip
    :return: argv list for the compressor
    """
    if shutil.which("bgzip"):
        return ["bgzip", "-@", str(threads), "-c"]
    # bgzip should always be present, but never fail QC over a missing binary
    return ["gzip"]


def chopper(
    input_long_reads, outdir, min_length, min_quality, gzip_flag, threads, logdir
):
    """Filters long reads using chopper

    :param input_long_reads: input ONT reads file
    :param outdir: output directory
    :param min_length: minimum length for long reads - defaults to 1000
    :param min_quality:  minimum quality for long reads - defaults to 8
    :param gzip_flag: whether or not the long reads are gzipped
    :param logdir
    :return:
    """
    filtered_long_reads: Path = Path(outdir) / "chopper_long_reads.fastq.gz"
    logger.info("Started running chopper")
    logdir.mkdir(parents=True, exist_ok=True)
    tool_name = Path("chopper").name
    logfile_prefix: Path = logdir / f"{tool_name}"
    chopper_cmd = [
        "chopper",
        "-q",
        min_quality,
        "--threads",
        threads,
        "-l",
        min_length,
        "--headcrop",
        "75",
        "--tailcrop",
        "75",
    ]
    compressor_cmd = gzip_compressor_cmd(threads)

    # ExitStack guarantees the log, output and pipe handles are closed even if a
    # Popen raises partway through building the chain
    with ExitStack() as stack:
        err_log = stack.enter_context(open(f"{logfile_prefix}.err", "w"))
        out_fh = stack.enter_context(open(filtered_long_reads, "wb"))

        stages = []
        try:
            if gzip_flag is True:
                source_proc = sp.Popen(
                    ["gunzip", "-c", input_long_reads], stdout=sp.PIPE, stderr=err_log
                )
                stages.append(("gunzip", source_proc))
                chopper_stdin = source_proc.stdout
            else:
                # plain fastq needs no decompressor: hand the file straight to
                # chopper rather than spawning a `cat` to copy it through a pipe
                chopper_stdin = stack.enter_context(open(input_long_reads, "rb"))

            chopper_proc = sp.Popen(
                chopper_cmd, stdin=chopper_stdin, stdout=sp.PIPE, stderr=err_log
            )
            stages.append(("chopper", chopper_proc))
            # the parent must drop its copy of each upstream read end, otherwise
            # the downstream stage never sees EOF
            if gzip_flag is True:
                source_proc.stdout.close()

            compress_proc = sp.Popen(
                compressor_cmd, stdin=chopper_proc.stdout, stdout=out_fh, stderr=err_log
            )
            stages.append((compressor_cmd[0], compress_proc))
            chopper_proc.stdout.close()
        except OSError as e:
            for _, proc in stages:
                proc.kill()
            logger.error(f"Error with chopper: {e}")
            return

        # every stage must be waited on. Previously only the last one was, so a
        # failing chopper was silently ignored and left a zombie behind
        failures = []
        for name, proc in reversed(stages):
            if proc.wait() != 0:
                failures.append(f"{name} (return code {proc.returncode})")

    if failures:
        logger.error(
            f"Error with chopper: {', '.join(reversed(failures))}. "
            f"Please check {logfile_prefix}.err"
        )
    logger.info("Finished running chopper")


def fastp(short_one, short_two, outdir, logdir):
    """Trims short reads using fastp

    :param short_one:  R1 short read file
    :param short_two:  R2 short read file
    :param outdir: output directory
    :param logger: logger
    :return:
    """
    outdir = Path(outdir)
    out_one: Path = outdir / "trimmed_R1.fastq"
    out_two: Path = outdir / "trimmed_R2.fastq"

    fastp = ExternalTool(
        tool="fastp",
        input=f"--in1 {short_one} --in2 {short_two}",
        output=f"--out1 {out_one} --out2 {out_two}",
        params="",
        logdir=logdir,
        outfile="",
    )

    ExternalTool.run_tool(fastp, to_stdout=False)


def copy_sr_fastq_file(infile: Path, outfile: Path):
    if infile.suffix == ".gz":
        # If the input file is a .fastq.gz file, extract and copy to .fastq
        with gzip.open(infile, "rt") as f_in:
            with open(outfile, "w") as f_out:
                f_out.writelines(f_in)
    elif infile.suffix == ".fastq":
        # If the input file is already a .fastq file, copy it directly
        shutil.copy2(infile, outfile)
    else:
        # Skip files that are not .fastq or .fastq.gz
        logger.error("Error with copy_sr_fastq_file")


def gzip_file(input_path, threads=1):
    """gzips a file, in parallel where bgzip is available

    Used by --skip_qc to compress the copied long reads. python's gzip module is
    both single threaded and slower than the gzip binary, which is a poor fit for
    a multi-GB ONT fastq; fall back to it only if spawning the compressor fails.

    :param input_path: file to compress
    :param threads: threads to give the compressor
    :return: path of the compressed file
    """
    input_path = Path(input_path)
    output_path = input_path.with_suffix(input_path.suffix + ".gz")

    try:
        with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            sp.run(
                gzip_compressor_cmd(threads), stdin=f_in, stdout=f_out, check=True
            )
        return output_path
    except (OSError, sp.CalledProcessError) as e:
        logger.warning(f"Falling back to python gzip for {input_path}: {e}")

    with open(input_path, "rb") as f_in:
        with gzip.open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return output_path
