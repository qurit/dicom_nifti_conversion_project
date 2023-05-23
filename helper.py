# Necessary Imports
import os
import pydicom
import pandas as pd
import gzip
import shutil
import json
import csv
import numpy as np
import SimpleITK as sitk
from datetime import datetime
import platform
import dateutil
import argparse
import textwrap
import sys
import nibabel as nib

# Constants
pet_dir_name = 'PET'
gt_dir_name = 'GT'
fi_ext = '.nii.gz'
sitk_path = './sitk.py'
ai4elife_dir_name = 'pet'
dict_name = 'dict'
dict_ext = '.txt'

a,b,c,d,e,f = ['a', 'b', 'c', 'd', 'e', 'f']
titles_dict = {'a' : 'dicom2nifti',
               'b' : 'dcm2niix',
               'c' : 'dcmstack',
               'd' : 'sitk',
               'e' : 'lifex',
               'f' : 'slicer'}

def bqml_to_suv(dcm_file: pydicom.FileDataset) -> float:
    """
    Calculates the conversion factor from Bq/mL to SUV bw [g/mL] using
    the dicom header information in one of the images from a dicom series
    """
    nuclide_dose = dcm_file[0x054, 0x0016][0][0x0018, 0x1074].value
    weight = dcm_file[0x0010, 0x1030].value
    half_life = float(dcm_file[0x054, 0x0016][0][0x0018, 0x1075].value) 
    
    
    parse = lambda x: dateutil.parser.parse(x)
    
    series_time = str(dcm_file[0x0008, 0x00031].value)
    series_date = str(dcm_file[0x0008, 0x00021].value) 
    series_datetime_str = series_date + ' ' + series_time
    series_dt = parse(series_datetime_str)
    
    nuclide_time = str(dcm_file[0x054, 0x0016][0][0x0018, 0x1072].value)
    nuclide_datetime_str = series_date + ' ' + nuclide_time
    nuclide_dt = parse(nuclide_datetime_str)
    
    
    delta_time = (series_dt - nuclide_dt).total_seconds()
    decay_correction = 2 ** (-1 * delta_time/half_life)
    suv_factor = (weight * 1000) / (decay_correction * nuclide_dose)
    Rescale_Slope= dcm_file[0x0028,0x1053].value
    Rescale_Intercept=dcm_file[0x0028,0x1052].value
    
    
    return (suv_factor, Rescale_Slope, Rescale_Intercept)


def check_input_dir(input_dir):
    """
    Check whether the provided input directory has provided the files
    using the convention provided in the git repository
    """
    # Check the input directory and see if the directories exists
    if not(os.path.isdir(input_dir)):
        raise SystemExit("\nProvided input directory does not exist")

    # Obtain a list of all of our cases (paths)
    cases=[]
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if (os.path.isdir(path)):
            cases.append(path)
    # If no cases, exit
    if not(cases):
        raise SystemExit("\nInput directory is empty")

    # For all the cases, check that they have the PET and GT dirs
    for case in cases:
        names= os.listdir(case)
        for type in [pet_dir_name, gt_dir_name]:
            if not(type in names):
                raise SystemExit(f"\n{case} does not have {type} directory")
            type_path = os.path.join(case, type)
            if not(os.listdir(type_path)):
                raise SystemExit(f"\n{case} has empty {type} directory")
            
    sys.stdout.write(f"\nInput Directory is Suitable")

def create_output_dir(output_dir):
    """
    Create output directory if necessary
    """
    if not(os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    sys.stdout.write(f"Output directory at {output_dir}")


def a_conv(output_path, pet_dir):
    """
    dicom2nifti conversion and file renaming. Provides path to file
    """
    a_exe = f'dicom2nifti {pet_dir} {output_path}'
    os.system(a_exe)
    # Renaming the file
    fi = os.listdir(output_path)[0]
    fi_path = os.path.join(output_path, titles_dict[a]+fi_ext)
    os.rename(os.path.join(output_path, fi), fi_path)

    sys.stdout.write(f"{titles_dict[a]} complete\n")

    return fi_path
            
def b_conv(output_path, pet_dir):
    """
    dcm2niix conversion. Provides path to file
    """
    b_exe = f'dcm2niix -z y -f {titles_dict[b]} -o {output_path} {pet_dir} 1> /dev/null'
    os.system(b_exe)
    
    fi_name = titles_dict[b]+fi_ext
    fi_path = os.path.join(output_path, fi_name)

    sys.stdout.write(f"{titles_dict[b]} complete\n")

    return fi_path

def c_conv(output_path, pet_dir):
    """
    dcmstack conversion
    """
    c_exe = f'dcmstack -d --output-ext {fi_ext} --dest-dir {output_path} -o {titles_dict[c]} {pet_dir}'
    os.system(c_exe)
    
    fi_name = titles_dict[c]+fi_ext
    fi_path = os.path.join(output_path, fi_name)
    sys.stdout.write(f"{titles_dict[c]} complete\n")

    return fi_path


def d_conv(output_path, pet_dir):
    """
    sitk conversion
    """
    d_exe = f'python {sitk_path} -i {pet_dir} -o {output_path} -f {titles_dict[d]}'
    os.system(d_exe)
    sys.stdout.write(f"{titles_dict[d]} complete")

def get_suv(pet_dir):
    """
    Goes through the PET directory, selects a single file and obtains
    the necessary values for the conversion
    """
    reader = sitk.ImageSeriesReader()
    seriesNames = reader.GetGDCMSeriesFileNames(pet_dir)
    # Extract a single DICOM file for the SUV calculation
    pet = pydicom.dcmread(seriesNames[0])
    # Calculate the SUV values
    suv_result = bqml_to_suv(pet)
    suv_factor = suv_result[0]
    Rescale_Slope = suv_result[1]
    Rescale_Intercept = suv_result[2]
    return suv_factor, Rescale_Slope, Rescale_Intercept

def scale_nifti(fi_path, suv_factor, Rescale_Slope, Rescale_Intercept):
    """
    Given the NIfTI file name, will scale it to transform from 
    bq/mL to SUV. This is for dicom2nifti, dcm2niix, dcmstack
    """
    img = nib.load(fi_path)
    header = img.header
    header.set_slope_inter(Rescale_Slope*suv_factor, Rescale_Intercept*suv_factor)
    header.get_slope_inter()
    nib.save(img, fi_path)

def make_conv_dir(output_path, name, key):
    """
    Make the directory in the form accessible by ai4elife
    """
    dir_name = name+'_'+titles_dict[key]
    dir_path = os.path.join(output_path, dir_name)
    os.mkdir(dir_path)
    dir_path = os.path.join(dir_path, ai4elife_dir_name)
    os.mkdir(dir_path)

    return dir_path

def make_dict(names_, output_dir):
    """
    Given all the names, will make the relevant dictionary
    """
    dict_path = os.path.join(output_dir, dict_name+dict_ext)
    file = open(dict_path,"w")
    names = [] 
    for name_ in names_:
        name = name_+'\n'
        names.append(name)
    file.writelines(names)
    file.close()


def file_conversion(input_dir, output_dir, do_a, do_b, do_c, do_d, do_e, do_f):
    """
    Execute the appropriate DICOM to NIfTI conversions as requested
    """
    names = []
    dirs = os.listdir(input_dir)
    no_dirs = len(dirs)
    for i,name in enumerate(dirs):
        path = os.path.join(input_dir, name)
        # The directories will correspond to individual patients
        try:
            if (os.path.isdir(path)):
                names.append(name)
                sys.stdout.write("\n"+f"-"*100+ "\n")
                sys.stdout.write(f"{i+1}/{no_dirs}: Working on {name}"+ "\n")
                pet_dir = os.path.join(path, pet_dir_name)
                if (do_a or do_b or do_c):
                    suv_factor, Rescale_Slope, Rescale_Intercept = get_suv(pet_dir)
                if do_a:
                    a_dir = make_conv_dir(output_dir, name, a)
                    a_path=a_conv(a_dir, pet_dir)
                    scale_nifti(a_path, suv_factor, Rescale_Slope, Rescale_Intercept)
                if do_b:
                    b_dir = make_conv_dir(output_dir, name, b)
                    b_path=b_conv(b_dir, pet_dir)
                    scale_nifti(b_path, suv_factor, Rescale_Slope, Rescale_Intercept)
                if do_c:
                    c_dir = make_conv_dir(output_dir, name, c)
                    c_path=c_conv(c_dir, pet_dir)
                    scale_nifti(c_path, suv_factor, Rescale_Slope, Rescale_Intercept)
                if do_d:
                    d_dir = make_conv_dir(output_dir, name, d)
                    d_conv(d_dir, pet_dir)
                if do_e:
                    _ = make_conv_dir(output_dir, name, e)
                if do_f:
                    _ = make_conv_dir(output_dir, name, f)
        except:
            raise SystemError(f"{name} failed")
    make_dict(names, output_dir)


