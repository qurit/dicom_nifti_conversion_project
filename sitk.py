#!/usr/bin/env python3

from helper import *

# Arguments to be passed
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_dir", help="path to dir with DICOM series folders", type=str, required=True)
argParser.add_argument("-o", "--output_dir", help="path to dir where NIfTI file will be saved", type=str, required=True)
argParser.add_argument("-f", "--file_name", help="name of the nifti file (without extension)", type=str, required=True)
args = argParser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
file_name = args.file_name

dicomToNifti(input_dir=input_dir, output_dir=output_dir, file_name=file_name)
