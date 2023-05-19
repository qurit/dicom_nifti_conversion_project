#!/usr/bin/env python3

# Necessary Imports
import argparse
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

# Arguments to be passed
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_dir", help="path to dir with DICOM series folders", type=str, required=True)
argParser.add_argument("-o", "--output_dir", help="path to dir where NIfTI file will be saved", type=str, required=True)
argParser.add_argument("-f", "--file_name", help="name of the nifti file (without extension)", type=str, required=True)
args = argParser.parse_args()

def bqml_to_suv(dcm_file: pydicom.FileDataset) -> float:
    '''
    Calculates the conversion factor from Bq/mL to SUV bw [g/mL] using 
    the dicom header information in one of the images from a dicom series
    '''
    nuclide_dose = dcm_file[0x054, 0x0016][0][0x0018, 0x1074].value  # Total injected dose (Bq)
    weight = dcm_file[0x0010, 0x1030].value  # Patient weight (Kg)
    half_life = float(dcm_file[0x054, 0x0016][0][0x0018, 0x1075].value)  # Radionuclide half life (s)

    parse = lambda x: dateutil.parser.parse(x)

    series_time = str(dcm_file[0x0008, 0x00031].value)  # Series start time (hh:mm:ss)
    series_date = str(dcm_file[0x0008, 0x00021].value)  # Series start date (yyy:mm:dd)
    series_datetime_str = series_date + ' ' + series_time
    series_dt = parse(series_datetime_str)

    nuclide_time = str(dcm_file[0x054, 0x0016][0][0x0018, 0x1072].value)  # Radionuclide time of injection (hh:mm:ss)
    nuclide_datetime_str = series_date + ' ' + nuclide_time
    nuclide_dt = parse(nuclide_datetime_str)

    delta_time = (series_dt - nuclide_dt).total_seconds()
    decay_correction = 2 ** (-1 * delta_time/half_life)
    suv_factor = (weight * 1000) / (decay_correction * nuclide_dose)
    Rescale_Slope= dcm_file[0x0028,0x1053].value
    Rescale_Intercept=dcm_file[0x0028,0x1052].value

    return (suv_factor, Rescale_Slope, Rescale_Intercept)

def dicomToNifti(input_dir, output_dir, file_name):
    # converts DICOM series in the seriesDir to NIFTI image in the savePath specified
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.dcm'):
                filename = os.path.join(root, file)
                ds = pydicom.dcmread(filename, force=True)
                break

    reader = sitk.ImageSeriesReader()
    seriesNames = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(seriesNames)
    image = reader.Execute()
    
    pet = pydicom.dcmread(seriesNames[0])  # read one of the images for header info
    suv_result = bqml_to_suv(pet)
    suv_factor = suv_result[0]
    Rescale_Slope = suv_result[1]
    Rescale_Intercept = suv_result[2]

    image = sitk.Multiply(image, Rescale_Slope)
    image = image + Rescale_Intercept
    image = sitk.Multiply(image, suv_factor)

    sitk.WriteImage(image, os.path.join(output_dir, f'{file_name}.nii.gz'), imageIO='NiftiImageIO')

input_dir = args.input_dir
output_dir = args.output_dir
file_name = args.file_name

dicomToNifti(input_dir=input_dir, output_dir=output_dir, file_name=file_name)
