#!/usr/bin/env python3

import argparse
import textwrap
import sys
import os
import nibabel as nib

from all_constants import *

def get_nifti_fi_path(case, input_dir):
    """
    Get the path of the nifti file
    """
    case_path = os.path.join(input_dir, case)
    pet_path = os.path.join(case_path, ai4elife_pet_dir_name)
    for fi in os.listdir(pet_path):
        if fi.endswith(fi_ext):
            nifti_fi = fi
    nifti_fi_path = os.path.join(pet_path, nifti_fi)

    return nifti_fi_path

def check_conv_type(case):
    """
    Return the conversion type
    """
    for key in keys:
        if case.endswith(titles_dict[key]):
            conv_type = titles_dict[key]
            break
    return conv_type

def extract_img_affine(nifti_fi_path):
    """
    Get the image (as ndarray) and the affine
    """
    img = nib.load(nifti_fi_path)
    affine  = img.affine
    img = img.get_fdata()

    return img, affine

def coordinate_fis(input_dir):
    """
    Load and save each nifti file (in pet dir) using
    nibabel and apply the mirroring (to coordinate
    all of them) if required
    """
    cases = os.listdir(input_dir)
    N = len(cases)
    for n,case in enumerate(cases):
        nifti_fi_path = get_nifti_fi_path(case, input_dir)
        conv_type = check_conv_type(case)
        img, affine = extract_img_affine(nifti_fi_path)
        if (mir_required[conv_type]):
            img = img[:, ::-1, :]
        output_img = nib.Nifti1Image(img, affine)
        nib.save(output_img, nifti_fi_path)
        sys.stdout.write(f"{n+1}/{N}"+"\n")

argParser = argparse.ArgumentParser(
    prog='PROG',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        Necessary coordination of the NIfTI Files
        '''))
argParser.add_argument("-i", "--input_dir", help="path to dir with patient nifti file directories", type=str, required=True)

args = argParser.parse_args()

sys.stdout.write("\n"+f"-"*100+"\n")

input_dir = args.input_dir
coordinate_fis(input_dir)