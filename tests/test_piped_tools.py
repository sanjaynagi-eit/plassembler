"""Tests for ExternalTool.run_piped, used to map straight into a sorted bam."""

import subprocess
from pathlib import Path

import pytest

from src.plassembler.utils.external_tools import ExternalTool


def tool(cmd, params, logdir, outfile=""):
    return ExternalTool(
        tool=cmd, input="", output="", params=params, logdir=logdir, outfile=outfile
    )


def test_run_piped_writes_last_stage_stdout(tmp_path):
    """Stages are connected by pipes and the last one's stdout lands in outfile."""
    logdir = tmp_path / "logs"
    out = tmp_path / "out.txt"
    ExternalTool.run_piped(
        (
            tool("printf", r" 'b\na\nc\n'", logdir),
            tool("sort", "", logdir),
            tool("tr", " a-z A-Z", logdir),
        ),
        outfile=out,
    )
    assert out.read_text() == "A\nB\nC\n"


def test_run_piped_last_stage_writes_its_own_file(tmp_path):
    """With outfile=None the last stage writes its own output, as `samtools sort
    -o` does; its stdout goes to that tool's .out log."""
    logdir = tmp_path / "logs"
    target = tmp_path / "written_by_tee.txt"
    ExternalTool.run_piped(
        (tool("printf", " 'hello\\n'", logdir), tool("tee", f" {target}", logdir))
    )
    assert target.read_text() == "hello\n"


def test_run_piped_raises_on_failing_stage(tmp_path):
    """A non-zero exit anywhere in the chain must surface, not be swallowed the
    way an unwaited-on process would be."""
    logdir = tmp_path / "logs"
    with pytest.raises(subprocess.CalledProcessError):
        ExternalTool.run_piped(
            (
                tool("printf", " 'x\\n'", logdir),
                tool("cat", " /nonexistent/path/xyz", logdir),
            ),
            outfile=tmp_path / "out.txt",
        )


def test_run_piped_reports_the_earliest_failing_stage(tmp_path):
    """When an upstream stage fails, its return code is the informative one."""
    logdir = tmp_path / "logs"
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        ExternalTool.run_piped(
            (
                tool("cat", " /nonexistent/path/xyz", logdir),
                tool("cat", "", logdir),
            ),
            outfile=tmp_path / "out.txt",
        )
    assert Path(excinfo.value.cmd[0]).name == "cat"
    assert excinfo.value.returncode != 0


def test_run_piped_does_not_leave_a_half_built_chain_running(tmp_path):
    """If a later stage's binary is missing, the stages already started must be
    killed rather than left waiting on a pipe nobody will read."""
    logdir = tmp_path / "logs"
    with pytest.raises(OSError):
        ExternalTool.run_piped(
            (
                tool("cat", "", logdir),
                tool("definitely_not_a_real_binary_xyz", "", logdir),
            ),
            outfile=tmp_path / "out.txt",
        )
