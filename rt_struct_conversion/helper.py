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
from rt_utils import RTStructBuilder
import platform
import dateutil
import argparse
import textwrap
import sys
import nibabel as nib
import matplotlib.pyplot as plt
import math
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import warnings
from rich.progress import track
warnings.filterwarnings("ignore")

from all_constants import *

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
    sys.stdout.write(f"Output directory at {output_dir}\n")
    for gt_key in gt_keys:
        key_dir = os.path.join(output_dir, gt_dict[gt_key])
        if not(os.path.isdir(key_dir)):
            os.mkdir(key_dir)
        temp_dir = os.path.join(key_dir, temp_dir_name)
        if not(os.path.isdir(temp_dir)):
            os.mkdir(temp_dir)
    


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

    return fi_path
            
def b_conv(output_path, pet_dir):
    """
    dcm2niix conversion. Provides path to file
    """
    b_exe = f'dcm2niix -z y -f {titles_dict[b]} -o {output_path} {pet_dir} 1> /dev/null'
    os.system(b_exe)
    
    fi_name = titles_dict[b]+fi_ext
    fi_path = os.path.join(output_path, fi_name)

    return fi_path

def c_conv(output_path, pet_dir):
    """
    dcmstack conversion
    """
    c_exe = f'dcmstack -d --output-ext {fi_ext} --dest-dir {output_path} -o {titles_dict[c]} {pet_dir}'
    os.system(c_exe)
    
    fi_name = titles_dict[c]+fi_ext
    fi_path = os.path.join(output_path, fi_name)

    return fi_path

def dicomToNifti(input_dir, output_dir, file_name):
    """
    converts DICOM series in the seriesDir to NIFTI image
    in the output_dir specified
    """
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
    
    # read one of the images for header info
    pet = pydicom.dcmread(seriesNames[0])
    suv_result = bqml_to_suv(pet)
    suv_factor = suv_result[0]
    Rescale_Slope = suv_result[1]
    Rescale_Intercept = suv_result[2]

    image = sitk.Multiply(image, Rescale_Slope)
    image = image + Rescale_Intercept
    image = sitk.Multiply(image, suv_factor)

    sitk.WriteImage(image,
                    os.path.join(output_dir, f'{file_name}.nii.gz'),
                    imageIO='NiftiImageIO')

def d_conv(output_path, pet_dir):
    """
    sitk conversion
    """
    d_exe = f'python {sitk_path} -i {pet_dir} -o {output_path} -f {titles_dict[d]}'
    os.system(d_exe)

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

def make_conv_dir(output_path, case, key):
    """
    Make the directory in the form accessible by ai4elife
    """
    dir_name = case+'_'+titles_dict[key]
    dir = os.path.join(output_path, dir_name)
    os.mkdir(dir)
    pet_dir = os.path.join(dir, ai4elife_pet_dir_name)
    os.mkdir(pet_dir)
    return dir, pet_dir

def make_gt_dirs(dirs):
    """
    Make the gt dirs in all the conversion dirs
    """
    gt_dirs = []
    for dir in dirs:
        gt_dir = os.path.join(dir, ai4elife_gt_dir_name)
        os.mkdir(gt_dir)
        gt_dirs.append(gt_dir)
    return gt_dirs

def buildMaskArray(fi, seriesPath, labelPath) -> np.ndarray:
    """
    Helper for the following function: taken from rt_utils
    """
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=seriesPath, rt_struct_path=labelPath)
    
    rois = rtstruct.get_roi_names()
    good_rois = []
    
    for roi in rois:
        if not(bad_word_is_in_roi(roi)):
            good_rois.append(roi)
    
    masks = []
    for roi in good_rois:
        mask_3d = rtstruct.get_roi_mask_by_name(roi).astype(int)
        masks.append(mask_3d)

    final_mask = sum(masks)  # sums element-wise
    final_mask = np.where(final_mask>=1, 1, 0)
    # Reorient the mask to line up with the reference image
    final_mask = np.moveaxis(final_mask, [0, 1, 2], [1, 2, 0])

    return final_mask

def buildMasks(file, seriesPath, path_to_gt_file, output_dir, case):
    """
    To convert the gt to the correct .nii.gz file
    """
    final_mask = buildMaskArray(file, seriesPath, path_to_gt_file)
    # SUV_* BP Liver
    # Load original DICOM image for reference
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(seriesPath)
    reader.SetFileNames(dicom_names)
    ref_img = reader.Execute()

    # Properly reference and convert the mask to an image object
    mask_img = sitk.GetImageFromArray(final_mask)
    mask_img.CopyInformation(ref_img)
    fi_name = case+'_gt'+fi_ext
    sitk.WriteImage(mask_img, os.path.join(output_dir, fi_name), imageIO="NiftiImageIO")
    return fi_name

def dcm2nii_mask(gt_file, gt_output_dir, fi_name, a_file):
    """
    Conversion of the gt to a NIfTI file
    """
    # conversion of the mask dicom file to nifti (not directly possible with dicom2nifti)
    mask = pydicom.read_file(gt_file)
    mask_array = mask.pixel_array
    
    # get mask array to correct orientation (this procedure is dataset specific)
    mask_array = np.transpose(mask_array,(2,1,0) )  
    mask_orientation = mask[0x5200, 0x9229][0].PlaneOrientationSequence[0].ImageOrientationPatient
    if mask_orientation[4] == 1:
        mask_array = np.flip(mask_array, 1 )
    
    # get affine matrix from the corresponding pet          
    pet = nib.load(a_file)
    pet_affine = pet.affine
    
    # return mask as nifti object
    mask_out = nib.Nifti1Image(mask_array, pet_affine)
    nib.save(mask_out, os.path.join(gt_output_dir, fi_name))
    
def y_gt_conv():
    """
    Pyradise gt conversion
    """
    return ""

def bad_word_is_in_roi(roi):
        truth = False
        for bad_word in bad_words:
            truth = truth or (bad_word in roi)
        return truth

def combine_rois(rois, dir_0, fi_name):
    good_rois = []
    for roi in rois:
        if not(bad_word_is_in_roi(roi)):
            good_rois.append(roi)
    rois = good_rois
    ref_roi = rois[0]
    ref_roi_path = os.path.join(dir_0, ref_roi)
    ref_mask = nib.load(ref_roi_path)
    rois.remove(ref_roi)
    affine = ref_mask.affine
    header = ref_mask.header

    final_mask = ref_mask.get_fdata()

    for roi in rois:
        roi_path = os.path.join(dir_0, roi)
        mask = nib.load(roi_path).get_fdata()
        final_mask += mask

    img = nib.Nifti1Image(final_mask, affine=affine, header=header)
    nib.save(img, os.path.join(dir_0, fi_name))
    
    for roi in rois:
        roi_path = os.path.join(dir_0, roi)
        os.remove(roi_path)
    

def z_gt_conv(gt_file_path, pet_dir, dir_0, case):
    """
    dcmrtstruct2nii conversion
    """
    z_exe = f"dcmrtstruct2nii convert -c False -r {gt_file_path}  -d {pet_dir} -o {dir_0} > /dev/null 2>&1"
    os.system(z_exe)
    
    rois = os.listdir(dir_0)
    if not(len(rois)==1):
        fi_name = case+"_gt"+fi_ext
        combine_rois(rois, dir_0, fi_name)
    else:
        fi_name = rois[0]

    return fi_name

def get_gt_fi(dir_0, case, lifex):
    """
    lifex and slicer conversion conversion
    """
    if lifex:
        gt_dir = lifex_gt_path
    else:
        gt_dir = slicer_gt_path
    
    gt_case_dir = os.path.join(gt_dir, case)
    
    gt_fis = os.listdir(gt_case_dir)
    gt_fi = gt_fis[0]
    
    shutil.copy(os.path.join(gt_case_dir, gt_fi),
                os.path.join(dir_0, gt_fi))
    
    return gt_fi
    

def gt_conv(case_dir, gt_dirs, case, a_path, rt_key, pet_dir):
    """
    ground truth conversion and saves to all the conversion directories appropriately
    """
    dir_0 = gt_dirs[0]
    pet_dir = os.path.join(case_dir, pet_dir_name)
    gt_dir = os.path.join(case_dir, gt_dir_name)
    gt_file_name = os.listdir(gt_dir)[0]
    gt_file_path = os.path.join(gt_dir, gt_file_name)
    
    if not(case.startswith(no_rt_struct_prefix)):
        match rt_key:
            case "x":
                fi_name=buildMasks(pydicom.dcmread(gt_file_path, force=True),
                                   pet_dir, gt_file_path, dir_0, case)
            # case y:
            #     fi_name=y_gt_conv()
            case "z":
                fi_name=z_gt_conv(gt_file_path, pet_dir, dir_0, case)
            case "u":
                fi_name=get_gt_fi(dir_0, case, lifex=True)
            case "v":
                fi_name=get_gt_fi(dir_0, case, lifex=False)
    else:
        fi_name = case+'_gt'+fi_ext
        dcm2nii_mask(gt_file_path, dir_0, fi_name, a_path)
    
    n=len(gt_dirs)
    for m in np.arange(1, n):
        shutil.copy(os.path.join(dir_0, fi_name), os.path.join(gt_dirs[m], fi_name))

def make_dict(cases_, output_dir):
    """
    Given all the cases, will make the relevant dictionary
    """
    dict_path = os.path.join(output_dir, dict_name+dict_ext)
    file = open(dict_path,"w")
    cases = [] 
    for case_ in cases_:
        case = case_+'\n'
        cases.append(case)
    file.writelines(cases)
    file.close()


def file_conversion(input_dir, output_dir, gt_key):
    """
    Execute the appropriate DICOM to NIfTI conversions as requested
    """
    cases = []
    dirs = os.listdir(input_dir)
    for case in track(dirs, description="Making NIfTI files..."):
        input_path = os.path.join(input_dir, case)
        key_dir = os.path.join(output_dir, gt_dict[gt_key])
        key_temp_dir = os.path.join(key_dir, temp_dir_name)
        # The directories will correspond to individual patients
        try:
            if (os.path.isdir(input_path)):
                cases.append(case)
                pet_dir = os.path.join(input_path, pet_dir_name)
                suv_factor, Rescale_Slope, Rescale_Intercept = get_suv(pet_dir)

                # a
                a_dir, a_pet_dir = make_conv_dir(key_temp_dir, case, a)
                a_path=a_conv(a_pet_dir, pet_dir)
                scale_nifti(a_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                # b 
                b_dir, b_pet_dir = make_conv_dir(key_temp_dir, case, b)
                b_path=b_conv(b_pet_dir, pet_dir)
                scale_nifti(b_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                # c
                c_dir, c_pet_dir = make_conv_dir(key_temp_dir, case, c)
                c_path=c_conv(c_pet_dir, pet_dir)
                scale_nifti(c_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                # d
                d_dir, d_pet_dir = make_conv_dir(key_temp_dir, case, d)
                d_conv(d_pet_dir, pet_dir)

                # e
                e_dir, _ = make_conv_dir(key_temp_dir, case, e)
                
                # f
                f_dir, _ = make_conv_dir(key_temp_dir, case, f)

                # gt
                gt_dirs = make_gt_dirs([a_dir, b_dir, c_dir, d_dir, e_dir, f_dir])
                gt_conv(input_path, gt_dirs, case, a_path, gt_key, pet_dir)
        except:
            raise SystemError(f"{case} failed")
    make_dict(cases, key_temp_dir)
    
    return key_dir, cases
    
def create_fis(input_dir, output_dir):
    """
    Create all of the relevant nifti files
    """
    sys.stdout.write("\n"+f"-"*100)
    # Check that the provided input directory is suitable
    check_input_dir(input_dir)
    sys.stdout.write("\n"+f"-"*100+ "\n")
    # Create output directory if necessary
    create_output_dir(output_dir)
    # Do the file conversions
    key_dirs = []
    n=0
    for gt_key in gt_keys:
        if (n==0):
            key_dir, cases=file_conversion(input_dir, output_dir, gt_key)
        else:
            key_dir, _=file_conversion(input_dir, output_dir, gt_key)
        key_dirs.append(key_dir)
        n+=1
    sys.stdout.write("\n"+f"-"*100)
    return key_dirs, cases

def satisfactory_lifex_slicer_dir(lifex_slicer_dir, cases):
    """ 
    Ensure that the directory has the relevant directories for each case
    """
    for case in cases:
        lifex_dir = os.path.join(lifex_slicer_dir, case+"_"+titles_dict[e])
        if not (os.path.isdir(lifex_dir)):
            raise SystemExit(f"No lifex directory for {case}")
        if not (os.listdir(lifex_dir)):
            raise SystemExit(f"lifex directory empty for {case}")
        slicer_dir = os.path.join(lifex_slicer_dir, case+"_"+titles_dict[f])
        if not (os.path.isdir(slicer_dir)):
            raise SystemExit(f"No slicer directory for {case}")
        if not (os.listdir(slicer_dir)):
            raise SystemExit(f"slicer directory empty for {case}")
        

def move_lifex_slicer_fis(key_dirs, lifex_slicer_dir):
    """
    Move these manually created slicer and lifex files into
    their appropriate directories
    """
    for key_dir in key_dirs:
        temp_dir = os.path.join(key_dir, temp_dir_name)
        dict_path = os.path.join(temp_dir, dict_name+dict_ext)
        fi = open(dict_path, 'r')
        _cases = fi.readlines()
        cases = []
        for _case in _cases:
            case = _case.replace('\n', '')
            cases.append(case)
        convs = [titles_dict[e], titles_dict[f]]
        for case in track(cases, description="Moving Lifex and Slicer Files..."):
            for conv in convs:
                old_conv_path = os.path.join(lifex_slicer_dir, case+'_'+conv, 'pet')
                new_conv_path = os.path.join(temp_dir, case+'_'+conv, 'pet')
                fi = os.listdir(old_conv_path)[0]
                shutil.copy(os.path.join(old_conv_path, fi),
                            os.path.join(new_conv_path, fi))
    sys.stdout.write("\n"+f"-"*100)
    
def process_lifex_slicer_fis(key_dirs, lifex_slicer_dir, cases):
    """ 
    Check that the provided dir has the relevant 
    """
    satisfactory_lifex_slicer_dir(lifex_slicer_dir, cases)
    move_lifex_slicer_fis(key_dirs, lifex_slicer_dir)
    
def get_nifti_fi_path(case_path):
    """
    Get the path of the nifti file
    """
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
    for case in track(os.listdir(input_dir), description="Coordinating Files..."):
        case_path = os.path.join(input_dir, case)
        if os.path.isdir(case_path):
            nifti_fi_path = get_nifti_fi_path(case_path)
            conv_type = check_conv_type(case)
            img, affine = extract_img_affine(nifti_fi_path)
            if (mir_required[conv_type]):
                img = img[:, ::-1, :]
            output_img = nib.Nifti1Image(img, affine)
            nib.save(output_img, nifti_fi_path)

def coordinate_dir(output_dir):
    """
    Coordinate all the appropriate files in all the temp dirs
    """
    sys.stdout.write("\n"+f"-"*100)
    for gt_key in gt_keys:
        gt_dir_path = os.path.join(output_dir, gt_dict[gt_key])
        gt_temp_dir = os.path.join(gt_dir_path, temp_dir_name)
        coordinate_fis(gt_temp_dir)
    sys.stdout.write(f"-"*100+"\n")
      

def read_dict(input_dir):
    """
    Given the path to the directory holding our dictionary, provide a 
    list of all the cases
    """
    dict_path = os.path.join(input_dir, dict_name+dict_ext)

    file = open(dict_path,"r")
    _cases = file.readlines()
    cases = []
    for _case in _cases:
        case = _case.replace('\n', "")
        cases.append(case)

    return cases

def get_dirs(input_dir):
    """
    Given the temporary directory path, provide a list of all
    the directory names
    """
    pred_dir = os.path.join(input_dir, pred_dir_name)
    dirs = os.listdir(pred_dir)
    return dirs


def get_combos(n):
    """
    Given a number, provide all combinations between indices less than
    said number
    """
    if (n==0) or (n==1):
        raise SystemError('n too low')
    combos = []
    for x in np.arange(n):
        for y in np.arange(x+1, n,1):
            combos.append([x,y])
    return combos

def get_case_data(dirs, case, input_dir):
    """
    Run through all the directories and get the relevant images for our case
    """
    case_data = []

    for dir in dirs:
        if dir.startswith(case):
            conv_type = dir.replace(case+"_", "")
            path = os.path.join(input_dir, pred_dir_name, dir)
            for fi in os.listdir(path):
                fi_path = os.path.join(path, fi)
                if fi.endswith(pet_end):
                    pet_img = nib.load(fi_path)
                elif fi.endswith(pred_end):
                    mask_img = nib.load(fi_path)
                elif fi.endswith(gt_end):
                    gt_img = nib.load(fi_path)
            case_data.append([conv_type, pet_img, mask_img, gt_img])
    return case_data

def get_vol_data(case, temp_dir):
    """
    Get all the volume datas relevant to this case
    """
    vol_data = []
    for dir in os.listdir(temp_dir):
        if dir.startswith(case):
            dir_path = os.path.join(temp_dir, dir)
            conv_type = dir.replace(case+"_", "")
            pet_path = os.path.join(dir_path, ai4elife_pet_dir_name)
            fis = os.listdir(pet_path)
            for fi in fis:
                if fi.endswith(fi_ext):
                    pet_fi = fi
            fi_path = os.path.join(pet_path, pet_fi)
            img = nib.load(fi_path)
            vol_data.append([conv_type, img])
    return vol_data        

def get_differences(dict):
    """
    For an images of arbitrary (but equal) dimensions, compute the absolute
    differences between all values and provide them as a 1D list
    """
    n = len(dict)
    combos = get_combos(n)
    differences = []
    for combo in combos:
        x,y = combo
        _ = []
        for u,v in zip(dict[titles_dict[keys[x]]], dict[titles_dict[keys[y]]]):
            _.append(abs(u-v))
        differences.append(_)
    return differences

def get_mask_matrix(diffs, case_dir):
    """
    Given all the data, provided a confusion matrix of the mask values
    """
    vals = np.array([
    [0, sum(diffs[0]), sum(diffs[1]), sum(diffs[2]), sum(diffs[3]), sum(diffs[4])],
    [0,0, sum(diffs[5]), sum(diffs[6]), sum(diffs[7]), sum(diffs[8])], 
    [0,0,0, sum(diffs[9]), sum(diffs[10]), sum(diffs[11])],
    [0,0,0,0, sum(diffs[12]), sum(diffs[13])], 
    [0,0,0,0,0, sum(diffs[14])], 
    [0,0,0,0,0,0]
    ])

    plt.rcParams["figure.figsize"] = [8, 7]
    plt.rcParams["figure.autolayout"] = True
    mask = np.tri(vals.shape[0], k=-1)
    data = np.ma.array(vals, mask=mask)
    plt.imshow(data, interpolation="nearest", cmap='viridis', extent=[-1, 1, -1, 1])
    plt.title("Mask Confusion Matrix", fontsize=fontsize)
    plt.colorbar()
    for (x,y),label in np.ndenumerate(vals):
        if x<=y:
            plt.text((-5+2*y)/6, (5-2*x)/6,label,ha='center',va='center')
    plt.xticks(ticks = [-5/6, -3/6, -1/6, 1/6, 3/6, 5/6],
               labels = [titles_dict[a], titles_dict[b], titles_dict[c], titles_dict[d], titles_dict[e], titles_dict[f]])
    plt.yticks(ticks = [-5/6, -3/6, -1/6, 1/6, 3/6, 5/6],
               labels = [titles_dict[f], titles_dict[e], titles_dict[d], titles_dict[c], titles_dict[b], titles_dict[a]])
    plt.savefig(os.path.join(case_dir, 'mask_vals.png'))
    plt.clf()

def get_titles():
    """
    Get all titles for the histogram/subtracted plots
    """
    combos = get_combos(len(keys))

    titles = []
    for combo in combos:
        x,y = combo
        title = f"{titles_dict[keys[x]]} vs. {titles_dict[keys[y]]}"
        titles.append(title)
    return titles

def get_hist(diffs, case_dir):
    """
    Make histograms for the comparisons
    """
    
    # Constants
    xmin = -12
    xmax = 2
    _bins = np.arange(xmin, xmax, 1)
    bins = []
    for _bin in _bins:
        bin =math.pow(10, _bin)
        bins.append(bin)
    no_rows = 3
    no_cols = 5
    
    fig, axs = plt.subplots(no_rows,no_cols, figsize=(22,16), sharey=True, sharex=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)

    fig.suptitle('Conversion Method Differences', fontsize=fontsize, y = 0.999)
    fig.supylabel('Instances', fontsize=fontsize, x=0.01)
    fig.supxlabel('Absolute Difference', fontsize=fontsize, y=0.01)

    vals = np.array(diffs).reshape(no_rows, no_cols, -1)
    titles = get_titles()
    titles = np.array(titles).reshape(no_rows, no_cols)

    for m in np.arange(no_rows):
        for n in np.arange(no_cols):
            axs[m][n].hist(vals[m][n], bins)
            axs[m][n].semilogx()
            axs[m][n].set_title(titles[m][n], fontsize=16)
            axs[m][n].set_xlim(xmin, xmax)
    
    fig.savefig(os.path.join(case_dir, 'histogram.png'))
    plt.clf()

def get_cut_differences(dict, cut):
    """
    Given a specific cut: {0:sagittal, 1:coronal}, provide all
    of the subtracted images. 
    """
    n = len(dict)
    combos = get_combos(n)
    cut_subs = []
    for combo in combos:
        cut_datas = []
        for ix in combo:
            img = dict[titles_dict[keys[ix]]]
            cut_data = img.get_fdata()[cut, :, :]
            cut_datas.append(cut_data)
        cut_sub = abs(cut_datas[0]-cut_datas[1])
        cut_subs.append(cut_sub)
    return cut_subs


def get_sub_img(cut_subs, case_dir, cut):
    """
    Given the cut differences (for all conversion types),
    provide the 
    """
    # Constants
    no_rows = 3
    no_cols = 5
    cmap=cm.get_cmap('viridis')
    #normalizer=Normalize(0,0.001)
    im=cm.ScalarMappable(cmap=cmap)
    cut_x, cut_y, cut_z = cut_subs[0].shape

    vals = np.array(cut_subs).reshape(no_rows, no_cols, cut_x, cut_y, cut_z)
    titles = get_titles()
    titles = np.array(titles).reshape(no_rows, no_cols)

    x_size = 18
    y_size = 16

    fig, axs = plt.subplots(no_rows,no_cols, figsize=(x_size,y_size), sharey=True, sharex=True, layout='constrained')

    for m in np.arange(no_rows):
        for n in np.arange(no_cols):
            data = vals[m][n]
            # Just to orient them properly (head up)
            data = np.swapaxes(data,1,0)
            data = data[::-1, :, :]
            axs[m][n].imshow(data, cmap=cmap)
            axs[m][n].set_title(titles[m][n], size=18)
            axs[m][n].axis('off')
    fig.colorbar(im, ax=axs[:, n], location='right')
    plt.savefig(os.path.join(case_dir, f'{cut_dict[cut]}_subtracted_plots.png'))
    plt.clf()


def get_subtracted_plots(pet_img_dict, case_dir):
    """
    Make the subtracted plots for the given comparisons
    """
    cuts = [0,1]

    for cut in cuts:
        cut_subs = get_cut_differences(pet_img_dict, cut)
        get_sub_img(cut_subs, case_dir, cut)

def compute_dice(img1, img2, eps=1e-5):
    """
    Compute the dice score between two images where
    """
    img1_area = img1.sum()
    img2_area = img2.sum()
    intersection = np.logical_and(img1, img2).sum()
    dice = 2*intersection / (img1_area + img2_area+eps)
    return dice


def compute_dices(img1, img2):
    """
    Compute the two dice-scores of the two images

    cut_dict = {0:'Sagittal', 1: 'Coronal'}
    """
    scores = []
    for cut in cuts:
        img1_cut = img1.get_fdata()[cut, :, :]
        img2_cut = img2.get_fdata()[cut, :, :]
        cut_dice=compute_dice(img1_cut, img2_cut)
        scores.append(cut_dice)

    return scores

def compute_mae(key, vol_vals_dict):
    """
    Compute the mean absolute error vs. the sitk method
    """
    if key==d:
        return 0
    else:
        ref_vals = vol_vals_dict[titles_dict[d]]
        comp_vals = vol_vals_dict[titles_dict[key]]
        aes = []
        n = len(ref_vals)
        for (r,c) in zip(ref_vals,comp_vals):
            ae = abs(r-c)
            aes.append(ae)
        mae = sum(aes)/n
        return mae

def get_num_vals(gt_and_mask_dict, vol_vals_dict, case_dir, case, case_columns):
    """"
    Provided with the predicted mask and gt mask, compute the dice-score
    and mean absolute error and provide a .csv file
    """
    entries = []
    for key in keys:
        gt, mask = gt_and_mask_dict[titles_dict[key]]
        sag_dice_score, cor_dice_score = compute_dices(gt, mask)
        mean_abs_error =compute_mae(key, vol_vals_dict)
        entry = [case, titles_dict[key], sag_dice_score,
                 cor_dice_score, mean_abs_error]
        entries.append(entry)

    df = pd.DataFrame(entries, columns=case_columns)
    df.to_csv(os.path.join(case_dir, f"{case}.csv"), index=False)

def get_results(case_dir, case_data, vol_data, case):
    """
    Given all the data, provide a confusion matrix of the mask values,
    histogram of the pet values absolute differences and subtracted images
    """
    pet_vals_dict = {}
    pet_img_dict = {}
    gt_and_mask_dict = {}
    mask_vals_dict = {}
    for data in case_data:
        conv_type = data[0]
        pet_img = data[1]
        pet_img_dict[conv_type] = pet_img
        pet_val = pet_img.get_fdata().reshape(-1)
        pet_vals_dict[conv_type]=pet_val 
        mask_img = data[2]
        mask_val = mask_img.get_fdata().reshape(-1)
        mask_vals_dict[conv_type]=mask_val 
        gt_img = data[3]
        gt_and_mask_dict[conv_type] = [gt_img, mask_img]
    
    vol_vals_dict={}
    for data in vol_data:
        conv_type = data[0]
        vol_img = data[1]
        vol_img_vals = vol_img.get_fdata().reshape(-1)
        vol_vals_dict[conv_type] = vol_img_vals
        
    #pet_diffs = get_differences(pet_vals_dict)
    #mask_diffs = get_differences(mask_vals_dict)

    #get_hist(pet_diffs, case_dir)
    #get_mask_matrix(mask_diffs, case_dir)
    #get_subtracted_plots(pet_img_dict, case_dir)
    get_num_vals(gt_and_mask_dict, vol_vals_dict,
                 case_dir, case, case_columns)

def get_surrogate_dict(fi_keys, temp_dir, fis_dict):
    """
    Provide a dictionary with all the relevant data from the 
    surrogate .csv files
    """
    surrogate_data = {}

    for fi_key in fi_keys:
        df = pd.read_csv(os.path.join(temp_dir, fis_dict[fi_key]))
        pids = df.iloc[:, 0]
        sag_TMTVs = df.iloc[:, 1]
        cor_TMTVs = df.iloc[:, 2]
        tot_TMTVs = df.iloc[:, 3]
        entry = {PID: pids, SAG: sag_TMTVs,
                 COR: cor_TMTVs, TOT: tot_TMTVs}
        surrogate_data[fi_key] = entry
    return surrogate_data

def get_case_df_vals(output_dir, case):
    """
    Provided with the output dir and the case,
    get the relevant .csv file and obtain its entries
    """
    case_dir = os.path.join(output_dir, case)
    case_fis = os.listdir(case_dir)
    for fi in case_fis:
        if fi.endswith(csv_ext):
            case_df = pd.read_csv(os.path.join(case_dir, fi))
            case_df_vals = case_df.values.tolist()
    return case_df_vals

def get_case_entries(surrogate_data, case):
    """
    Given the surrogate_data dictionary and the case,
    get all the relevant entries's indices and 
    conversion types
    """
    case_entries = []
    for ix, pid in enumerate(surrogate_data[GT][PID]):
        if pid.startswith(case):
            conv_type = pid.replace(case+"_", "")
            case_entries.append([ix, conv_type])
    return case_entries

def combine_case_data(case_entries, surrogate_data, case_df_vals):
    """
    Combine the case_df_vals and surrogate_data (at
    the correct indices) into entries for our final 
    .csv file
    """
    combined_vals = []
    for ix, conv_type in case_entries:
        pred_sag_TMTV= surrogate_data[PRED][SAG][ix]
        pred_cor_TMTV= surrogate_data[PRED][COR][ix]
        pred_tot_TMTV= surrogate_data[PRED][TOT][ix]
        gt_sag_TMTV= surrogate_data[GT][SAG][ix]
        gt_cor_TMTV= surrogate_data[GT][COR][ix]
        gt_tot_TMTV= surrogate_data[GT][TOT][ix]
        for case_df_val in case_df_vals:
            if (case_df_val[1]== conv_type):
                case_df_val.extend([pred_sag_TMTV, gt_sag_TMTV,
                                    pred_cor_TMTV, gt_cor_TMTV,
                                    pred_tot_TMTV, gt_tot_TMTV])
                combined_vals.append(case_df_val)
    return combined_vals

def get_dict_vals(cases, output_dir, surrogate_data):
    """
    Get all the entries for our final .csv file
    """
    dict_vals = []
    for case in cases:
        # Get relevant info
        case_df_vals = get_case_df_vals(output_dir, case)
        case_entries = get_case_entries(surrogate_data, case)
        # Combine them accordingly
        entry = combine_case_data(case_entries, surrogate_data,
                                  case_df_vals)
        dict_vals.extend(entry)
    return dict_vals


def make_result_file(temp_dir, output_dir, cases):
    """
    Make the final .csv file with the entries containing the case,
    conversion type, dice scores and TMTV values. 

    Will take values from the output_dir (dice scores from the .csv files)
    and the temp_dir (with the surrogate .csv files). 
    """
    fis_dict = {GT: gt_csv_fi, PRED: pred_csv_fi}

    surrogate_data = get_surrogate_dict(fi_keys, temp_dir, fis_dict)

    dict_vals = get_dict_vals(cases, output_dir, surrogate_data)

    final_df = pd.DataFrame(dict_vals, columns = all_columns)
    final_df.to_csv(os.path.join(output_dir, results_fi_name), index=False)
    
def combine_result_files(input_dir):
    all_data_vals = []
    for gt_key in gt_keys:
        gt_name = gt_dict[gt_key]
        gt_results_fi = os.path.join(input_dir, gt_name, results_dir_name, results_fi_name)
        gt_results = pd.read_csv(gt_results_fi)
        gt_results = gt_results.values.tolist()
        new_entries = []
        for entry in gt_results:
            if entry[1].startswith(no_rt_struct_prefix):
                name = no_rt_struct_conv_name
            else:
                name = gt_name
            entry = [name]+entry
            new_entries.append(entry)
        all_data_vals.extend(new_entries)
    combined_df = pd.DataFrame(all_data_vals, columns = combined_columns)
    combined_df.to_csv(os.path.join(input_dir, combined_results_fi), index=False)
    
def get_abs_errs(lst_of_two_entry_lists):
    """ 
    Given n entries each containing two lists, compute the absolute
    differences (errors) between the two list and provide all these
    in the same order
    """
    abs_errs = []
    for two_entry in lst_of_two_entry_lists:
        lsta = two_entry[0]
        lstb = two_entry[1]
        
        lstab_abs_errs = []
        for a,b in zip (lsta, lstb):
            abs_err = abs(a-b)
            lstab_abs_errs.append(abs_err)
        abs_errs.append(lstab_abs_errs)
    return abs_errs
        
def get_gt_result_dict(gt_convs, sag_dices, cor_dices, sag_abs_errs, cor_abs_errs, tot_abs_errs):
    """
    For each gt conversion method, provide the sag_dice_scores, 
    cor_dice_scores, sag_tmtvs, cor_tmtvs, tot_tmtvs in that order
    """
    gt_result_dict = {}
    conv_names = []
    sag_dice_lst, cor_dice_lst, sag_tmtv_lst, cor_tmtv_lst, tot_tmtv_lst = [],[],[],[],[]
    results_lsts = [sag_dice_lst, cor_dice_lst, sag_tmtv_lst,
                    cor_tmtv_lst, tot_tmtv_lst]
    data_lsts = [sag_dices, cor_dices, sag_abs_errs,
                 cor_abs_errs, tot_abs_errs]
    data_ixs = [SAG_DICE, COR_DICE, SAG_TMTV, COR_TMTV, TOT_TMTV]
    
    for gt_key in gt_keys:
        conv_name = gt_dict[gt_key]
        conv_names.append(conv_name)
        for results_lst, data_lst in zip(results_lsts, data_lsts):
            scores = []
            for ix, conv_type in enumerate(gt_convs):
                if (conv_type==conv_name):
                    scores.append(data_lst[ix])
            results_lst.append(scores)
    gt_result_dict[CONV_TYPE] = conv_names
    for data_ix, result_lst in zip(data_ixs, results_lsts):
        gt_result_dict[data_ix] = result_lst
    
    return gt_result_dict
           
def make_plots(data, names, input_dir, dice, cut): 
    """ 
    Make the appropriate boxplots
    """
    plt.boxplot(data)
    plt.xticks([1,2,3,4], names)
    cut_name = cut_dict[cut]
    if dice:
        title = f"{cut_name} Dice Scores"
        box_plot_y_axis = "Dice Score"
        fi_name = "Dice"
    else:
        title = f"{cut_name} TMTV"
        box_plot_y_axis = "TMTV ($mm^2$)"
        fi_name = "TMTV"
    plt.title(title)
    plt.rcParams['figure.figsize']=(10,10)
    plt.xlabel(box_plot_x_axis)
    plt.ylabel(box_plot_y_axis)
    plt.savefig(os.path.join(input_dir,
                             cut_name+"_"+fi_name+"_"+box_plot_fi_name))
    plt.clf()

def make_dice_boxplots(gt_data_dict, input_dir):
    """
    Provide the box plots for the four gt conversion methods
    """
    make_plots(gt_data_dict[SAG_DICE], gt_data_dict[CONV_TYPE],
               input_dir, dice=True, cut=0)
    make_plots(gt_data_dict[COR_DICE], gt_data_dict[CONV_TYPE],
               input_dir, dice=True, cut=1)
    
def make_tmtv_boxplot(gt_data_dict, input_dir):
    """
    Provide the box plots for the four gt conversion methods
    """
    make_plots(gt_data_dict[SAG_TMTV], gt_data_dict[CONV_TYPE],
               input_dir, dice=False, cut=0)
    make_plots(gt_data_dict[COR_TMTV], gt_data_dict[CONV_TYPE],
               input_dir, dice=False, cut=1)
    make_plots(gt_data_dict[TOT_TMTV], gt_data_dict[CONV_TYPE],
               input_dir, dice=False, cut=2)
    
def make_comparison_table(gt_data_dict, input_dir):
    """
    Compute all the mean average values and store them in .csv file
    """
    entries = []

    for ix in np.arange(len(gt_data_dict[CONV_TYPE])):
        entry = []
        for data_type in [SAG_DICE, COR_DICE, SAG_TMTV, COR_TMTV, TOT_TMTV]:
            entry_elem = np.mean(gt_data_dict[SAG_DICE][ix])
            entry.append(entry_elem)
        entries.append(entry)
                 
    df = pd.DataFrame(entries, columns = gt_columns)
    df.to_csv(os.path.join(input_dir, gt_csv_fi), index=False)
    
def get_overall_results(input_dir):
     """ 
     Provide the box plots and table to
     illustrate the mean absolute errors
     """   
     result_fi_path = os.path.join(input_dir, combined_results_fi)
     results = pd.read_csv(result_fi_path)
     
     gt_convs = results[gt_conv_csv_index]
     sag_dices, cor_dices =results[sag_dice_csv_index], results[cor_dice_csv_index]
     pred_sag_tmtvs, gt_sag_tmtvs = results[pred_sag_tmtv_csv_index], results[gt_sag_tmtv_csv_index]
     pred_cor_tmtvs, gt_cor_tmtvs = results[pred_cor_tmtv_csv_index], results[gt_cor_tmtv_csv_index]
     pred_tot_tmtvs, gt_tot_tmtvs = results[pred_tot_tmtv_csv_index], results[gt_tot_tmtv_csv_index]
     
     sag_abs_errs, cor_abs_errs, tot_abs_errs = get_abs_errs([ [pred_sag_tmtvs, gt_sag_tmtvs], [pred_cor_tmtvs, gt_cor_tmtvs], [pred_tot_tmtvs, gt_tot_tmtvs]])
     
     gt_result_dict = get_gt_result_dict(gt_convs, sag_dices, cor_dices, sag_abs_errs, cor_abs_errs, tot_abs_errs)    
     make_dice_boxplots(gt_result_dict, input_dir)
     make_tmtv_boxplot(gt_result_dict, input_dir)
     make_comparison_table(gt_result_dict, input_dir)