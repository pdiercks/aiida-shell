# -*- coding: utf-8 -*-
"""AiiDA plugin that makes running shell commands easy."""
from .calculations import ShellJob
from .engine import launch_shell_job, shellfunction
from .parsers import ShellParser

__version__ = '0.1.0'
