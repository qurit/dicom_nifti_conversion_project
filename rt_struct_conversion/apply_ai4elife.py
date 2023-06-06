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
        '''))
argParser.add_argument("-i", "--input_dir", help="path to dir with all the 4 ground truths temp data", type=str, required=True)
argParser.add_argument("-a", "--ai_dir", help="path to dir with ai4elife (downloaded from github)", type=str, required=True)

args = argParser.parse_args()

sys.stdout.write("\n"+f"-"*100+"\n")

input_dir = args.input_dir
input_dir = os.path.abspath(input_dir)
ai_dir = args.ai_dir
ai_dir = os.path.abspath(ai_dir)

cwd = os.getcwd()
cwd = os.path.abspath(cwd)


for gt_key in gt_keys:
    gt_name = gt_dict[gt_key]
    sys.stdout.write(f"\nWorking on {gt_name}\n")
    gt_dir = os.path.join(input_dir, gt_name)
    gt_temp_dir = os.path.join(gt_dir, temp_dir_name)
    gt_ai4elife_dir = os.path.join(gt_dir, ai4elife_dir_name)
    os.mkdir(gt_ai4elife_dir)
    gt_ai4elife_dir = os.path.abspath(gt_ai4elife_dir)
    gt_temp_dir = os.path.abspath(gt_temp_dir)
    # Move the dictionary to the output directory
    dict_path = os.path.join(gt_temp_dir, dict_name+dict_ext)
    move_exe = f"mv {dict_path} {gt_ai4elife_dir}"
    os.system(move_exe)
    # Go to the ai4elife directory
    os.chdir(ai_dir)
    # Run the ai4elife command
    ai_exe = f"python test_env.py --input_dir {gt_temp_dir} --output_dir {gt_ai4elife_dir} 1> /dev/null"
    os.system(ai_exe)
    # Return to start
    os.chdir(cwd)