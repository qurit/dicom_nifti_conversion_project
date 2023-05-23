#!/usr/bin/env python3

from helper import *

# Arguments to be passed
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_dir", help="path to dir with DICOM series folders", type=str, required=True)
argParser.add_argument("-o", "--output_dir", help="path to dir where NIfTI file will be saved", type=str, required=True)
argParser.add_argument("-f", "--file_name", help="name of the nifti file (without extension)", type=str, required=True)
args = argParser.parse_args()

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
