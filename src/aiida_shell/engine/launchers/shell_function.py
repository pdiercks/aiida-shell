# -*- coding: utf-8 -*-
"""Convenience wrapper function to simplify the interface to launch a ``shellfunction``."""
from .functions import shellfunction

__all__ = ('shell_function',)


def shell_function(command, arguments, files, filenames=None, outputs=None):
    """Convenience wrapper function to simplify the interface to launch a ``shellfunction``."""
