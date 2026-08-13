import hashlib
import shlex
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional, Tuple

import click
from loguru import logger

"""
to make the BLAST dbs

 makeblastdb -in dnaA.faa -dbtype prot -out dnaA_db
 makeblastdb -in terL.faa -dbtype prot -out terL_db
 makeblastdb -in repA.faa -dbtype prot -out repA_db
"""


class ExternalTool:
    def __init__(
        self,
        tool: str,
        input: str,
        output: str,
        params: str,
        logdir: Path,
        outfile: Path,
    ):
        self.command: List[str] = self._build_command(tool, input, output, params)
        logdir.mkdir(parents=True, exist_ok=True)
        command_hash = hashlib.sha256(self.command_as_str.encode("utf-8")).hexdigest()
        tool_name = Path(tool).name
        # to make sure no spaces or -
        tool_name_with_underscores = tool_name.replace(" ", "_")
        tool_name_with_underscores = tool_name_with_underscores.replace("-", "_")
        logfile_prefix: Path = logdir / f"{tool_name_with_underscores}_{command_hash}"
        self.out_log = f"{logfile_prefix}.out"
        self.err_log = f"{logfile_prefix}.err"
        self.outfile = outfile
        self.tool_str = tool

    @property
    def command_as_str(self) -> str:
        return shlex.join(self.command)

    @staticmethod
    def _build_command(tool: str, input: str, output: str, params: str) -> List[str]:
        # note: shlex.join does not allow us to shlex.split() later
        # this is explicitly a " ".join()
        command = " ".join([tool, params, output, input])
        escaped_command = shlex.split(command)
        return escaped_command

    def run(self) -> None:
        with open(self.out_log, "w") as stdout_fh, open(self.err_log, "w") as stderr_fh:
            print(f"Command line: {self.command_as_str}", file=stderr_fh)
            logger.info(f"Started running {self.command_as_str} ...")
            self._run_core(self.command, stdout_fh=stdout_fh, stderr_fh=stderr_fh)
            logger.info(f"Done running {self.command_as_str}")

    def run_to_stdout(
        self,
    ) -> None:
        with open(self.outfile, "w") as outfile, open(self.err_log, "w") as stderr_fh:
            print(f"Command line: {self.command_as_str}", file=stderr_fh)
            logger.info(f"Started running {self.command_as_str} ...")
            self._run_core(self.command, stdout_fh=outfile, stderr_fh=stderr_fh)
            logger.info(f"Done running {self.command_as_str}")

    @staticmethod
    def _run_core(command: List[str], stdout_fh, stderr_fh) -> None:
        subprocess.check_call(command, stdout=stdout_fh, stderr=stderr_fh)

    @staticmethod
    def run_piped(
        tools: Tuple["ExternalTool", ...], outfile: Optional[Path] = None
    ) -> None:
        """Run several tools as one pipeline, without a shell.

        Used to avoid materialising intermediates that exist only so the next
        command can read them back - most importantly the full uncompressed SAM
        that each mapping step used to write to disk before samtools re-read it.

        :param tools: stages, in order; each stage's stdout feeds the next
        :param outfile: file for the last stage's stdout. If None the last stage
            writes its own output (e.g. `samtools sort -o`) and its stdout goes
            to that tool's .out log, matching ExternalTool.run.
        """
        joined = " | ".join(tool.command_as_str for tool in tools)
        logger.info(f"Started running {joined} ...")

        procs: List[subprocess.Popen] = []
        with ExitStack() as stack:
            if outfile is None:
                last_stdout = stack.enter_context(open(tools[-1].out_log, "w"))
            else:
                last_stdout = stack.enter_context(open(outfile, "wb"))

            upstream_stdout = None
            try:
                for index, tool in enumerate(tools):
                    stderr_fh = stack.enter_context(open(tool.err_log, "w"))
                    print(f"Command line: {tool.command_as_str}", file=stderr_fh)
                    is_last = index == len(tools) - 1
                    proc = subprocess.Popen(
                        tool.command,
                        stdin=upstream_stdout,
                        stdout=last_stdout if is_last else subprocess.PIPE,
                        stderr=stderr_fh,
                    )
                    # the parent must drop its copy of the upstream read end,
                    # otherwise that stage never sees EOF
                    if upstream_stdout is not None:
                        upstream_stdout.close()
                    upstream_stdout = proc.stdout
                    procs.append(proc)
            except OSError:
                # a stage failed to start (missing binary, fork failure). Do not
                # leave the earlier stages of a half-built chain running.
                for proc in procs:
                    proc.kill()
                    proc.wait()
                raise

            # wait downstream-first: an upstream stage blocked writing into a
            # dead pipe only gets its SIGPIPE once the reader is gone
            for proc in reversed(procs):
                proc.wait()

        # report the earliest failing stage, which is the informative one
        for tool, proc in zip(tools, procs):
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, tool.command)

        logger.info(f"Done running {joined}")

    @staticmethod
    def run_tools(
        tools_to_run: Tuple["ExternalTool", ...], ctx: Optional[click.Context] = None
    ) -> None:
        for tool in tools_to_run:
            try:
                tool.run()
            except subprocess.CalledProcessError as error:
                logger.error(
                    f"Error calling {tool.command_as_str} (return code {error.returncode})"
                )
                logger.error(f"Please check stdout log file: {tool.out_log}")
                logger.error(f"Please check stderr log file: {tool.err_log}")
                logger.error("Temporary files are preserved for debugging")
                logger.error("Exiting...")

                if ctx:
                    ctx.exit(1)
                else:
                    sys.exit(1)

    """
    Only one tool
    """

    @staticmethod
    def run_tool(
        tool: "ExternalTool",
        ctx: Optional[click.Context] = None,
        to_stdout: Optional[bool] = False,
    ) -> None:
        try:
            if to_stdout is False:
                tool.run()
            else:  # if tool needs to write to stdout
                tool.run_to_stdout()
        except subprocess.CalledProcessError as error:
            if "unicycler" in tool.tool_str:  # for unicycler errors
                logger.warning(
                    "Unicycler has failed. This usually means that you have no plasmids. Checking."
                )
            elif tool.tool_str == "dnaapler all":  # for dnaapler errors
                logger.warning(
                    "Dnaapler failed to reorient any putative plasmids to begin with repA."
                )
                logger.warning("Continuing with the un-reoriented contigs.")
            elif tool.tool_str == "canu":  # for canu errors
                logger.warning(
                    "Canu failed to assemble anything from the unmapped reads."
                )
                logger.warning(
                    f"If you think your sample should still have plasmids, please check stdout log file: {tool.out_log} and stderr log file: {tool.err_log}"
                )
            elif tool.tool_str == "canu -correct":  # for canu errors
                logger.warning("Canu failed to correct any reads.")
                logger.warning(
                    "This probably means there is low depth, don't be too concerned."
                )
                logger.warning(
                    f"If you are concerned, check stdout log file: {tool.out_log} and stderr log file: {tool.err_log}."
                )
            else:
                logger.warning(
                    f"Error calling {tool.command_as_str} (return code {error.returncode})"
                )
                logger.warning(f"Please check stdout log file: {tool.out_log}")
                logger.warning(f"Please check stderr log file: {tool.err_log}")
                logger.warning("Temporary files are preserved for debugging")
                logger.error("Exiting...")

                if ctx:
                    ctx.exit(1)
                else:
                    sys.exit(1)
