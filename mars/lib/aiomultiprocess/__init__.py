# Copyright 2018 John Reese
# Licensed under the MIT license

"""
AsyncIO version of the standard multiprocessing module
"""

__author__ = "John Reese"
__version__ = "0.7.0"

from .core import Process, Worker, set_context, set_start_method
from .pool import Pool
from .scheduler import RoundRobin, Scheduler
from .types import QueueID, TaskID
