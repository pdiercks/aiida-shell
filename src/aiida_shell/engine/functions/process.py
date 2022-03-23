# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Class and decorators to generate processes out of Python functions."""
import collections
import functools
import inspect
import logging
import signal
import typing as t

from aiida.common.lang import override
from aiida.engine.processes.process import Process
from aiida.manage.manager import get_manager
from aiida.orm import Data, ProcessNode
from aiida.orm.utils.mixins import FunctionCalculationMixin

if t.TYPE_CHECKING:
    from aiida.engine.processes.exit_code import ExitCode  # pylint: disable=unused-import

__all__ = ('FunctionProcess',)

LOGGER = logging.getLogger(__name__)


def process_function(node_class: t.Type['ProcessNode'],
                     process_class,
                     process_kwargs=None,
                     exit_codes=None) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
    """The base function decorator to create a ``FunctionProcess`` out of a normal Python function.

    :param node_class: the ORM class to be used as the node record for the ``FunctionProcess``.
    """

    def decorator(function: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        """Turn the decorated function into a ``FunctionProcess``.

        :param callable: the actual decorated function that the ``FunctionProcess`` represents.
        :return callable: The decorated function.
        """
        cls = process_class.build(function, node_class=node_class, exit_codes=exit_codes)

        def run_get_node(*args, **kwargs) -> t.Tuple[t.Optional[t.Dict[str, t.Any]], 'ProcessNode']:
            """Run the ``FunctionProcess`` with the supplied inputs in a local runner.

            :param args: input arguments to construct the ``FunctionProcess``.
            :param kwargs: input keyword arguments to construct the ``FunctionProcess``.
            :return: tuple of the outputs of the process and the process node.
            """
            manager = get_manager()
            runner = manager.get_runner()
            inputs = cls.create_inputs(*args, **kwargs)

            # Remove all the known inputs from the kwargs
            for port in cls.spec().inputs:
                kwargs.pop(port, None)

            # If any kwargs remain, the spec should be dynamic, so we raise if it isn't
            if kwargs and not cls.spec().inputs.dynamic:
                raise ValueError(f'{function.__name__} does not support these kwargs: {kwargs.keys()}')

            process = cls(inputs=inputs, runner=runner, **process_kwargs or {})

            # Only add handlers for interrupt signal to kill the process if we are in a local and not a daemon runner.
            # Without this check, running process functions in a daemon worker would be killed if the daemon is shutdown
            current_runner = manager.get_runner()
            original_handler = None
            kill_signal = signal.SIGINT

            if not current_runner.is_daemon_runner:

                def kill_process(_num, _frame):
                    """Send the kill signal to the process in the current scope."""
                    LOGGER.critical('runner received interrupt, killing process %s', process.pid)
                    result = process.kill(msg='Process was killed because the runner received an interrupt')
                    return result

                # Store the current handler on the signal such that it can be restored after process has terminated
                original_handler = signal.getsignal(kill_signal)
                signal.signal(kill_signal, kill_process)

            try:
                result = process.execute()
            finally:
                # If the `original_handler` is set, that means the `kill_process` was bound, which needs to be reset
                if original_handler:
                    signal.signal(signal.SIGINT, original_handler)

            store_provenance = inputs.get('metadata', {}).get('store_provenance', True)
            if not store_provenance:
                process.node._storable = False  # pylint: disable=protected-access
                process.node._unstorable_message = 'cannot store node because it was run with `store_provenance=False`'  # pylint: disable=protected-access

            return result, process.node

        def run_get_pk(*args, **kwargs) -> t.Tuple[t.Optional[t.Dict[str, t.Any]], int]:
            """Recreate the ``run_get_pk`` utility launcher.

            :param args: input arguments to construct the ``FunctionProcess``.
            :param kwargs: input keyword arguments to construct the ``FunctionProcess``.
            :return: tuple of the outputs of the process and the process node pk.
            """
            result, node = run_get_node(*args, **kwargs)
            return result, node.pk

        @functools.wraps(function)
        def decorated_function(*args, **kwargs):
            """This wrapper function is the actual function that is called."""
            result, _ = run_get_node(*args, **kwargs)
            return result

        decorated_function.run = decorated_function  # type: ignore[attr-defined]
        decorated_function.run_get_pk = run_get_pk  # type: ignore[attr-defined]
        decorated_function.run_get_node = run_get_node  # type: ignore[attr-defined]
        decorated_function.is_process_function = True  # type: ignore[attr-defined]
        decorated_function.node_class = node_class  # type: ignore[attr-defined]
        decorated_function.process_class = cls  # type: ignore[attr-defined]
        decorated_function.recreate_from = cls.recreate_from  # type: ignore[attr-defined]
        decorated_function.spec = cls.spec  # type: ignore[attr-defined]

        return decorated_function

    return decorator


class FunctionProcess(Process):
    """Class to represent simple functions as a ``Process``."""

    _func_args: t.Sequence[str] = ()

    @staticmethod
    def _func(*_args, **_kwargs) -> dict:
        """Execute the process that represents the wrapped function.

        This is a placeholder. The ``build`` method will replace it with the definition of the ``FunctionProcess`` which
        is built dynamically based on the signature of the Python function that is wrapped.
        """
        return {}

    @classmethod
    def build(cls,
              func: t.Callable[..., t.Any],
              node_class: t.Type['ProcessNode'],
              exit_codes=None) -> t.Type['FunctionProcess']:
        """Build the ``Process`` class whose spec is based on the signature of the wrapped function.

        All function arguments will be assigned as process inputs. If keyword arguments are specified then these will
        also become inputs.

        :param func: The function to build a process from.
        :param node_class: Provide a custom node class to be used, has to be constructable with no arguments. It has to
            be a sub class of `ProcessNode` and the mixin :class:`~aiida.orm.utils.mixins.FunctionCalculationMixin`.
        :return: A ``Process`` class that represents the function.
        """
        if not issubclass(node_class, ProcessNode) or not issubclass(node_class, FunctionCalculationMixin):
            raise TypeError('the node_class should be a sub class of `ProcessNode` and `FunctionCalculationMixin`')

        args, varargs, keywords, defaults, _, _, _ = inspect.getfullargspec(func)
        nargs = len(args)
        ndefaults = len(defaults) if defaults else 0
        first_default_pos = nargs - ndefaults

        if varargs is not None:
            raise ValueError('variadic arguments are not supported')

        def define(cls, spec):  # pylint: disable=unused-argument
            """Define the spec dynamically"""
            super().define(spec)

            for i, arg in enumerate(args):
                default = ()
                if defaults and i >= first_default_pos:
                    default = defaults[i - first_default_pos]

                # If the keyword was already specified, simply override the default
                if spec.has_input(arg):
                    spec.inputs[arg].default = default
                else:
                    # If the default is `None` make sure that the port also accepts a `NoneType`
                    # Note that we cannot use `None` because the validation will call `isinstance` which does not work
                    # when passing `None`, but it does work with `NoneType` which is returned by calling `type(None)`
                    if default is None:
                        valid_type = (Data, type(None))
                    else:
                        valid_type = (Data,)

                    spec.input(arg, valid_type=valid_type, default=default)

            # Set defaults for label and description based on function name and docstring, if not explicitly defined
            port_label = spec.inputs['metadata']['label']

            if not port_label.has_default():
                port_label.default = func.__name__

            # If the function support kwargs then allow dynamic inputs, otherwise disallow
            spec.inputs.dynamic = keywords is not None

            # Function processes must have a dynamic output namespace since we do not know beforehand what outputs
            # will be returned and the valid types for the value should be `Data` nodes as well as a dictionary because
            # the output namespace can be nested.
            spec.outputs.valid_type = (Data, dict)

            if exit_codes is not None:
                for status, key, message in exit_codes:
                    spec.exit_code(status, key, message)

        return type(
            func.__name__, (cls,), {
                '__module__': func.__module__,
                '__name__': func.__name__,
                '_func': staticmethod(func),
                Process.define.__name__: classmethod(define),
                '_func_args': args,
                '_node_class': node_class
            }
        )

    @classmethod
    def validate_inputs(cls, *args: t.Any, **kwargs: t.Any) -> None:  # pylint: disable=unused-argument
        """Validate the positional and keyword arguments passed in the function call.

        :raises TypeError: if more positional arguments are passed than the function defines.
        """
        nargs = len(args)
        nparameters = len(cls._func_args)

        # If the spec is dynamic, i.e. the function signature includes `**kwargs` and the number of positional arguments
        # passed is larger than the number of explicitly defined parameters in the signature, the inputs are invalid and
        # we should raise. If we don't, some of the passed arguments, intended to be positional arguments, will be
        # misinterpreted as keyword arguments, but they won't have an explicit name to use for the link label, causing
        # the input link to be completely lost.
        if cls.spec().inputs.dynamic and nargs > nparameters:
            name = cls._func.__name__
            raise TypeError(f'{name}() takes {nparameters} positional arguments but {nargs} were given')

    @classmethod
    def create_inputs(cls, *args: t.Any, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """Create the input args for the ``FunctionProcess``."""
        cls.validate_inputs(*args, **kwargs)

        ins = {}
        if kwargs:
            ins.update(kwargs)
        if args:
            ins.update(cls.args_to_dict(*args))
        return ins

    @classmethod
    def args_to_dict(cls, *args: t.Any) -> t.Dict[str, t.Any]:
        """Create an input dictionary (of form ``label -> value``) from supplied args.

        :param args: The values to use for the dictionary.
        :return: A label -> value dictionary.
        """
        return dict(list(zip(cls._func_args, args)))

    @classmethod
    def get_or_create_db_record(cls) -> 'ProcessNode':
        return cls._node_class()

    def __init__(self, *args, **kwargs) -> None:
        enable_persistence = kwargs.pop('enable_persistence', False)
        if enable_persistence:
            raise RuntimeError('Cannot persist a function process')
        super().__init__(*args, enable_persistence=False, **kwargs)  # type: ignore[misc]

    @property
    def process_class(self) -> t.Callable[..., t.Any]:
        """Return the class that represents this ``Process``, for the ``FunctionProcess`` this is the function itself.

        For a standard ``Process`` or sub class of ``Process``, this is the class itself. However, for legacy reasons,
        the ``Process`` class is a wrapper around another class. This function returns that original class, i.e. the
        class that really represents what was being executed.

        :return: A ``Process`` class that represents the function
        """
        return self._func

    def execute(self) -> t.Optional[t.Dict[str, t.Any]]:
        """Execute the process."""
        result = super().execute()

        # ``FunctionProcesses`` can return a single value as output, instead of a dictionary.
        if result and len(result) == 1 and self.SINGLE_OUTPUT_LINKNAME in result:
            return result[self.SINGLE_OUTPUT_LINKNAME]

        return result

    @override
    def _setup_db_record(self) -> None:
        """Set up the database record for the process."""
        super()._setup_db_record()
        self.node.store_source_info(self._func)

    def call_function(self, *args, **kwargs):
        """Call the wrapped function."""
        return self._func(*args, **kwargs)

    @override
    def run(self) -> t.Optional['ExitCode']:
        """Run the process."""
        # The following conditional is required for the caching to properly work. Even if the source node has a process
        # state of `Finished` the cached process will still enter the running state. The process state will have then
        # been overridden by the engine to `Running` so we cannot check that, but if the `exit_status` is anything other
        # than `None`, it should mean this node was taken from the cache, so the process should not be rerun.
        from aiida.engine.processes.exit_code import ExitCode  # pylint: disable=redefined-outer-name

        if self.node.exit_status is not None:
            return self.node.exit_status

        # Split the inputs into positional and keyword arguments
        args = [None] * len(self._func_args)
        kwargs = {}

        for name, value in (self.inputs or {}).items():
            try:
                if self.spec().inputs[name].non_db:  # type: ignore[union-attr]
                    # Don't consider non-database inputs
                    continue
            except KeyError:
                pass  # No port found

            # Check if it is a positional arg, if not then keyword
            try:
                args[self._func_args.index(name)] = value
            except ValueError:
                kwargs[name] = value

        result = self.call_function(*args, **kwargs)

        if result is None or isinstance(result, ExitCode):
            return result

        if isinstance(result, Data):
            self.out(self.SINGLE_OUTPUT_LINKNAME, result)
        elif isinstance(result, collections.abc.Mapping):
            for name, value in result.items():
                self.out(name, value)
        else:
            raise TypeError(
                "Function process returned an output with unsupported type '{}'\n"
                'Must be a Data type or a mapping of {{string: Data}}'.format(result.__class__)
            )

        return ExitCode()
