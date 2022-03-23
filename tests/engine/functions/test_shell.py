# -*- coding: utf-8 -*-
"""Tests for the :mod:`aiida_shell.engine.functions.shell` module."""
import datetime
import io

from aiida.orm import CalcFunctionNode, List, SinglefileData
import pytest

from aiida_shell import shellfunction


def test_error_command_not_found():
    """Test a shellfunction that is declaring a command that cannot be found."""

    @shellfunction(command='unknown-command')
    def command():
        """Run the ``date`` command."""

    _, node = command.run_get_node()
    assert node.is_failed
    assert node.exit_status == command.process_class.exit_codes.ERROR_COMMAND_NOT_FOUND.status


def test_error_command_failed():
    """Test that the shellfunction process fails if the command fails."""

    @shellfunction(command='tar')
    def command():
        """Run the ``tar`` command."""

    # Running ``tar`` without arguments should cause it to return a non-zero exit status.
    _, node = command.run_get_node()
    assert node.is_failed
    assert node.exit_status == command.process_class.exit_codes.ERROR_COMMAND_FAILED.status


def test_basic():
    """Test a shellfunction that is just a command."""

    @shellfunction(command='date')
    def command():
        """Run the ``date`` command."""

    results, node = command.run_get_node()
    assert results == {}
    assert isinstance(node, CalcFunctionNode)
    assert node.is_finished_ok
    assert sorted(node.list_object_names()) == ['source_file', 'stderr', 'stdout']
    assert node.get_object_content('stderr') == ''
    assert node.get_object_content('stdout') != ''


def test_attach_stdout():
    """Test a shellfunction that specifies the ``attach_stdout`` keyword.

    This argument should cause the stdout to be captured, wrapped in a ``SinglefileData`` node attached as an output.
    """

    @shellfunction(command='date', attach_stdout=True)
    def command():
        """Run the ``date`` command."""

    # Check that there is a ``SinglefileData`` result with label ``stdout`` whose content is non-empty
    results, node = command.run_get_node()
    assert node.is_finished_ok
    assert list(results.keys()) == ['stdout']
    assert isinstance(results['stdout'], SinglefileData)
    assert node.outputs.stdout.get_content() != ''


def test_positional_arguments():
    """Test a shellfunction that specifies positional CLI arguments."""

    @shellfunction(command='date')
    def command():
        """Run the ``date`` command."""

    # Have ``date`` print the current date without time in ISO 8601 format. Note that if we are very unlucky, the
    # shellfunction runs just before midnight and the comparison ``datetime`` call runs in the next day causing the test
    # to fail, but that seems extremely unlikely.
    arguments = List(['--iso-8601'])
    _, node = command.run_get_node(arguments=arguments)
    assert node.is_finished_ok
    assert node.get_object_content('stdout').strip() == datetime.datetime.now().strftime('%Y-%m-%d')


def test_keyword_arguments():
    """Test a shellfunction that specifies positional CLI arguments that are interpolated by the ``kwargs``."""

    @shellfunction(command='cat')
    def command():
        """Run the ``cat`` command."""

    content_a = 'content_a\n'
    content_b = 'content_b\n'
    file_a = SinglefileData(io.StringIO(content_a))
    file_b = SinglefileData(io.StringIO(content_b))
    arguments = List(['{file_a}', '{file_b}'])
    _, node = command.run_get_node(arguments=arguments, file_a=file_a, file_b=file_b)

    assert node.is_finished_ok
    assert node.get_object_content('stdout') == content_a + content_b


def test_mixed_arguments():
    """Test a shellfunction that specifies positional and keyword CLI arguments interpolated by the ``kwargs``."""

    @shellfunction(command='head')
    def command():
        """Run the ``head`` command."""

    content = 'line 1\nline 2'
    single_file = SinglefileData(io.StringIO(content))
    arguments = List(['-n', '1', '{single_file}'])
    _, node = command.run_get_node(arguments=arguments, single_file=single_file)

    assert node.is_finished_ok
    assert node.get_object_content('stdout').strip() == content.split('\n', maxsplit=1)[0]


def test_cwd():
    """Test a shellfunction that generates multiple output files and so we use the ``cwd`` magic variable."""

    @shellfunction(command='split')
    def command():
        """Run the ``split`` command."""
        results = {}

        # The ``cwd`` variable contains the path to the working directory where the shell command was executed.
        for file in cwd.iterdir():  # pylint: disable=undefined-variable
            # `split` writes output files with the format `xaa, xab, xac...` etc.
            if file.name.startswith('x'):
                results[file.name] = SinglefileData(file)

        return results

    content = 'line 0\nline 1\nline 2\n'
    single_file = SinglefileData(io.StringIO(content))
    arguments = List(['-l', '1', '{single_file}'])
    results, node = command.run_get_node(arguments=arguments, single_file=single_file)

    expected_keys = ['xaa', 'xab', 'xac']
    assert node.is_finished_ok
    assert sorted(results.keys()) == expected_keys
    for index, key in enumerate(expected_keys):
        assert isinstance(results[key], SinglefileData)
        assert results[key].get_content().strip() == f'line {index}'


def test_output_filenames():
    """Test the ``output_filenames`` shellfunction argument.

    This is the same test as ``test_cwd`` except that here the output capture is done by the engine and not implemented
    manually by the user in the shellfunction's body.
    """

    @shellfunction(command='split', output_filenames=['x*'])
    def command():
        """Run the ``split`` command."""

    content = 'line 0\nline 1\nline 2\n'
    single_file = SinglefileData(io.StringIO(content))
    arguments = List(['-l', '1', '{single_file}'])
    results, node = command.run_get_node(arguments=arguments, single_file=single_file)

    expected_keys = ['xaa', 'xab', 'xac']
    assert node.is_finished_ok
    assert sorted(results.keys()) == expected_keys
    for index, key in enumerate(expected_keys):
        assert isinstance(results[key], SinglefileData)
        assert results[key].get_content().strip() == f'line {index}'
