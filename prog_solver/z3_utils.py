import sys
sys.path.append('.')

import re
import hashlib
import logging

import os
from os.path import join
import subprocess
from subprocess import check_output
import signal

PREFIX = "tmp"
def hash_of_code(code, size=16):
    val = hashlib.sha1(code.encode("utf-8")).hexdigest()
    return val[-size:]

class ExecCache:
    cache = {}

def execute_z3_test(code, filename=None, flag_keepfile=False, timeout=1.0, use_cache=False):
    if filename is None:
        filename = hash_of_code(code)
        _use_cache = use_cache
    if _use_cache and filename in ExecCache.cache:
        return ExecCache.cache[filename]

    filename = join(PREFIX, filename + ".py")

    with open(filename, "w") as f:
        f.write(code)
    try:
        output = check_output(["python", filename], stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8").strip().splitlines()[-1]
        result = (False, "ExecutionError " + output)
        if use_cache:
            ExecCache.cache[filename] = result
        return result
    except subprocess.TimeoutExpired:
        result = (False, "TimeoutError")
        if use_cache:
            ExecCache.cache[filename] = result
        return result
    output = output.decode("utf-8").strip()
    if not flag_keepfile:
        os.remove(filename)
    if _use_cache:
        ExecCache.cache[filename] = (True, output)
    return (True,  output)

def make_z3_enum_line(sort_name, members):
    return "{}, ({}) = EnumSort('{}', [{}])".format(
        sort_name,
        ", ".join([f"{n}" for n in members]),
        sort_name,
        ", ".join([f"'{n}'" for n in members])
    )


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
