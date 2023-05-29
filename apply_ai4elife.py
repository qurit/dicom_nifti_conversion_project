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
argParser.add_argument("-i", "--input_dir", help="path to dir with patient nifti file directories", type=str, required=True)
argParser.add_argument("-o", "--output_dir", help="path to dir where processed NIfTI files will be saved", type=str, required=True)
argParser.add_argument("-a", "--ai_dir", help="path to dir with ai4elife (downloaded from github)", type=str, required=True)

args = argParser.parse_args()

sys.stdout.write("\n"+f"-"*100+"\n")

input_dir = args.input_dir
input_dir = os.path.abspath(input_dir)
output_dir = args.output_dir
ai_dir = args.ai_dir
ai_dir = os.path.abspath(ai_dir)

cwd = os.getcwd()
cwd = os.path.abspath(cwd)

# Make output directory if necessary
if not(os.path.isdir(output_dir)):
    os.mkdir(output_dir)
    output_dir = os.path.abspath(output_dir)

# Move the dictionary to the output directory
dict_path = os.path.join(input_dir, dict_name+dict_ext)
move_exe = f"mv {dict_path} {output_dir}"
os.system(move_exe)

# Go to the ai4elife directory
os.chdir(ai_dir)
# Run the ai4elife command
ai_exe = f"python test_env.py --input_dir {input_dir} --output_dir {output_dir} 1> /dev/null"
os.system(ai_exe)
# Return to start
os.chdir(cwd)