# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Module to transform a Python function into a process function that calls an arbitrary shell command."""
import io
import os
import pathlib
import shutil
import subprocess
import tempfile
import typing as t

from aiida.common.lang import override
from aiida.orm import CalcFunctionNode, List, SinglefileData

from .process import FunctionProcess, process_function

__all__ = ('shellfunction',)


def shellfunction(command, attach_stdout=False, output_filenames=None):
    """Turn a standard python function into a shellfunction."""

    def factory(function):
        """Construct the process function for the given arguments."""
        import inspect

        # Add ``kwargs`` to the signature of the function if it doesn't already specify it.
        signature = inspect.signature(function)

        if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            parameters = list(signature.parameters.values())
            parameters.append(inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD))
            signature = signature.replace(parameters=parameters)
            function.__signature__ = signature

        return process_function(
            node_class=CalcFunctionNode,
            process_class=ShellFunctionProcess,
            process_kwargs={
                'command': command,
                'attach_stdout': attach_stdout,
                'output_filenames': output_filenames,
            },
            exit_codes=(
                (250, 'ERROR_COMMAND_NOT_FOUND', 'The command `{command}` was not found.'),
                (251, 'ERROR_COMMAND_FAILED', 'The command returned a non-zero exit code: {exit_code}.'),
                (
                    252, 'ERROR_MISSING_OUTPUT_FILES',
                    'The command did not produce all declared output files: {missing_files}.'
                ),
            )
        )(function)

    return factory


class ShellFunctionProcess(FunctionProcess):
    """Subclass of ``FunctionProcess`` that overrides ``call_function`` to actually call the shell command."""

    def __init__(self, *args, **kwargs) -> None:
        self.command = kwargs.pop('command')
        self.attach_stdout = kwargs.pop('attach_stdout')
        self.output_filenames = kwargs.pop('output_filenames')
        super().__init__(*args, **kwargs)

        if self.output_filenames is not None and (
            not isinstance(self.output_filenames, list) or any(not isinstance(e, str) for e in self.output_filenames)
        ):
            raise TypeError(f'`output_filenames` should be a list of strings, got: {self.output_filenames}')

    @staticmethod
    def process_arguments(dirpath: pathlib.Path, arguments_node: List, kwargs) -> t.List[str]:
        """Process the command line arguments from the ``arguments_node`` input node.

        The ``arguments_node`` can contain strings that are placeholders. These should be replaced with the filepaths
        of files that are written to the temporary directory provided by ``dirpath``. The file content can be found in
        the ``kwargs`` where the placeholder name corresponds to the key and the value is a ``Singlefiledata`` node.

        :param dirpath: temporary folder to which the input files should be written.
        :param arguments_node: a ``List`` node containing input arguments.
        :return: list of command line arguments.
        """
        from string import Formatter

        if not isinstance(arguments_node, List):
            raise TypeError(
                f'the `arguments` input keyword argument should be a `List` node, but got: {type(arguments_node)}'
            )

        formatter = Formatter()
        processed = []

        for argument in arguments_node.get_list():

            # Parse the argument for placeholders.
            field_names = [name for _, name, _, _ in formatter.parse(argument) if name]

            # If the argument contains no placeholders simply append the argument and continue.
            if not field_names:
                processed.append(argument)
                continue

            # Otherwise we validate that there is exactly one placeholder and that a `SinglefileData` input node is
            # specified in the keyword arguments. This is written to the current working directory and the filename
            # is used to replace the placeholder.
            if len(field_names) > 1:
                raise ValueError(
                    f'command line argument `{argument}` is invalid as it contains more than one placeholder.'
                )

            field_name = field_names[0]

            if field_name not in kwargs:
                raise ValueError(
                    f'command line arguments include placeholder `{{{field_name}}}` but no corresponding file is '
                    'specified in the input keyword arguments.'
                )

            single_file = kwargs.pop(field_name)

            if not isinstance(single_file, SinglefileData):
                raise ValueError(
                    f'keyword argument `{{field_name}}` should be a `SinglefileData`, but got: {type(single_file)}.'
                )

            filename = f'{single_file.filename}'
            filepath = dirpath / filename

            with single_file.open(mode='rb') as handle:
                filepath.write_bytes(handle.read())

            argument_interpolated = argument.format(**{field_name: filepath})
            processed.append(argument_interpolated)

        return processed

    @override
    def call_function(self, *args, **kwargs):
        """Call the shell command."""
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches,too-many-nested-blocks
        command = self.command
        executable = shutil.which(command)

        if executable is None:
            return self.exit_codes.ERROR_COMMAND_NOT_FOUND.format(command=command)

        arguments = [executable]

        with tempfile.TemporaryDirectory() as tempdir:

            dirpath = pathlib.Path(tempdir)

            try:
                arguments_node = kwargs.pop('arguments')
            except KeyError:
                arguments_keyword = []
            else:
                arguments_keyword = self.process_arguments(dirpath, arguments_node, kwargs)

            arguments.extend(arguments_keyword)

            # Any remaining input arguments should now be ``SinglefileData`` nodes. They will be written to the working
            # directory and the filepath is added as a positional argument.
            for key, single_file in kwargs.items():
                if not isinstance(single_file, SinglefileData):
                    raise TypeError(
                        f'keyword argument `{key}` should be a `SinglefileData`, but got: {type(single_file)}'
                    )

                filepath = dirpath / f'{single_file.filename}'

                with single_file.open(mode='rb') as handle:
                    filepath.write_bytes(handle.read())

                arguments.append(str(filepath))

            # self.node.set_attribute('arguments', arguments)

            cwd = pathlib.Path.cwd()

            try:
                # Record current working directory and change to the temporary directory before execution
                os.chdir(dirpath)

                outputs = {}

                try:
                    result_subprocess = subprocess.run(arguments, capture_output=True, check=True)
                except subprocess.CalledProcessError as exception:
                    stderr = io.BytesIO(exception.stderr)
                    stdout = io.BytesIO(exception.output)
                    exit_code = self.exit_codes.ERROR_COMMAND_FAILED.format(exit_code=exception.returncode)
                else:
                    stderr = io.BytesIO(result_subprocess.stderr)
                    stdout = io.BytesIO(result_subprocess.stdout)
                    exit_code = None

                # Add the stdout and stderr to the file repository of the process node. The operation needs to be
                # performed directly on the ``_repository`` attribute of the node, because the public method will fail
                # due to the mutability check given that the node is already stored.
                # pylint: disable=protected-access
                self.node.base.repository._repository.put_object_from_filelike(stderr, 'stderr')

                # The stdout gets attached as an output node instead of written to the node's repository if the
                # ``attach_stdout`` option is set to ``True``.
                if self.attach_stdout:
                    outputs['stdout'] = SinglefileData(stdout, filename='stdout')
                else:
                    self.node.base.repository._repository.put_object_from_filelike(stdout, 'stdout')

                self.node.base.repository._update_repository_metadata()
                # pylint: enable=protected-access

                if exit_code:
                    return exit_code

                func_globals = self._func.__globals__  # type: ignore[attr-defined]
                saved_values = func_globals.copy()
                func_globals.update({'cwd': dirpath})
                try:
                    outputs_function = self._func(*args, **kwargs) or {}
                finally:
                    func_globals = saved_values

                missing_output_files = []

                if self.output_filenames:
                    for filename in self.output_filenames:
                        if '*' in filename:
                            for globbed in dirpath.glob(filename):
                                output_key = globbed.name.replace('.', '_')
                                if globbed.exists():
                                    outputs[output_key] = SinglefileData(globbed, filename=globbed.name)
                                else:
                                    missing_output_files.append(globbed.name)
                        else:
                            output_key = filename.replace('.', '_')
                            filepath = dirpath / filename
                            if filepath.exists():
                                outputs[output_key] = SinglefileData(filepath, filename=filename)
                            else:
                                missing_output_files.append(filepath.name)

                outputs.update(**outputs_function)

            finally:
                os.chdir(cwd)

            if missing_output_files:
                return self.exit_codes.ERROR_MISSING_OUTPUT_FILES.format(missing_files=', '.join(missing_output_files))

        return outputs
