#!/usr/bin/env python3

import argparse
import textwrap
import sys
import os

from all_constants import *

# Arguments to be passed
argParser = argparse.ArgumentParser(
    prog='PROG',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        Creation of send NIfTI files through ai4elife

        This must be run using the ai4elife specific environment
        --------------------------------
            The prompted main directory is the directory
            where the following directories are/will be saved:
            -data_dir
            -temp_dir
            -ai_dir
            -results_dir
            -lifex_slicer_dir
        '''))
argParser.add_argument("-m", "--main_dir", help="path to dir where all data/results will be saved", type=str, required=True)
argParser.add_argument("-a", "--ai_dir", help="path to dir with ai4elife (downloaded from github)", type=str, required=True)

args = argParser.parse_args()

sys.stdout.write("\n"+f"-"*100+"\n")

main_dir = args.main_dir
main_dir = os.path.abspath(main_dir)
ai_dir = args.ai_dir
ai_dir = os.path.abspath(ai_dir)

cwd = os.getcwd()
cwd = os.path.abspath(cwd)

temp_dir = os.path.join(main_dir, temp_dir_name)
ai_data_dir = os.path.join(main_dir, ai_dir_name)

# Make ai_data directory if necessary
if not(os.path.isdir(ai_data_dir)):
    os.mkdir(ai_data_dir)
    ai_data_dir = os.path.abspath(ai_data_dir)

# Move the dictionary to the ai_data directory
dict_path = os.path.join(temp_dir, dict_name+dict_ext)
move_exe = f"mv {dict_path} {ai_data_dir}"
os.system(move_exe)

# Go to the ai4elife directory
os.chdir(ai_dir)
# Run the ai4elife command
ai_exe = f"python test_env.py --input_dir {temp_dir} --output_dir {ai_data_dir} 1> /dev/null"
os.system(ai_exe)
# Return to start
os.chdir(cwd)