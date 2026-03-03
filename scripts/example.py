#!/usr/bin/env -S uv run --no-project --script
#SBATCH --account=aip-fredashi
#SBATCH --job-name=nothing
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

# /// script
# dependencies = ["sh"]
# ///

# A test script on how to write a Python Slurm script.

import json
import os
import sys
import traceback
from datetime import datetime

import sh

output = sh.python("-c", "print('hello world!')")
sh.tyw("tg", "text", output.strip(), _fg=True)