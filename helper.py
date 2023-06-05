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
from tqdm import tqdm
import warnings
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

def make_conv_dir(output_path, name, key):
    """
    Make the directory in the form accessible by ai4elife
    """
    dir_name = name+'_'+titles_dict[key]
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
    masks = []
    for roi in rois:
        mask_3d = rtstruct.get_roi_mask_by_name(roi).astype(int)
        masks.append(mask_3d)

    final_mask = sum(masks)  # sums element-wise
    final_mask = np.where(final_mask>=1, 1, 0)
    # Reorient the mask to line up with the reference image
    final_mask = np.moveaxis(final_mask, [0, 1, 2], [1, 2, 0])

    return final_mask

def buildMasks(file, seriesPath, path_to_gt_file, output_dir, name):
    """
    To convert the gt to the correct .nii.gz file
    """
    final_mask = buildMaskArray(file, seriesPath, path_to_gt_file)

    # Load original DICOM image for reference
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(seriesPath)
    reader.SetFileNames(dicom_names)
    ref_img = reader.Execute()

    # Properly reference and convert the mask to an image object
    mask_img = sitk.GetImageFromArray(final_mask)
    mask_img.CopyInformation(ref_img)
    fi_name = name+'_gt'+fi_ext
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

def gt_conv(case_dir, gt_dirs, name, a_path):
    """
    ground truth conversion and saves to all the conversion directories appropriately
    """
    dir_0 = gt_dirs[0]
    pet_dir = os.path.join(case_dir, pet_dir_name)
    gt_dir = os.path.join(case_dir, gt_dir_name)
    gt_file_name = os.listdir(gt_dir)[0]
    gt_file_path = os.path.join(gt_dir, gt_file_name)

    if not(name.startswith(no_rt_struct_prefix)):
        fi_name= buildMasks(pydicom.dcmread(gt_file_path, force=True), pet_dir, gt_file_path, dir_0, name)
    else:
        fi_name = name+'_gt'+fi_ext
        dcm2nii_mask(gt_file_path, dir_0, fi_name, a_path)

    n=len(gt_dirs)
    for m in np.arange(1, n):
        shutil.copy(os.path.join(dir_0, fi_name), os.path.join(gt_dirs[m], fi_name))

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


def file_conversion(input_dir, output_dir):
    """
    Execute the appropriate DICOM to NIfTI conversions as requested
    """
    names = []
    dirs = os.listdir(input_dir)
    for name in tqdm(dirs):
        input_path = os.path.join(input_dir, name)
        # The directories will correspond to individual patients
        try:
            if (os.path.isdir(input_path)):
                names.append(name)
                pet_dir = os.path.join(input_path, pet_dir_name)
                suv_factor, Rescale_Slope, Rescale_Intercept = get_suv(pet_dir)

                # a
                a_dir, a_pet_dir = make_conv_dir(output_dir, name, a)
                a_path=a_conv(a_pet_dir, pet_dir)
                scale_nifti(a_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                # b 
                b_dir, b_pet_dir = make_conv_dir(output_dir, name, b)
                b_path=b_conv(b_pet_dir, pet_dir)
                scale_nifti(b_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                # c
                c_dir, c_pet_dir = make_conv_dir(output_dir, name, c)
                c_path=c_conv(c_pet_dir, pet_dir)
                scale_nifti(c_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                # d
                d_dir, d_pet_dir = make_conv_dir(output_dir, name, d)
                d_conv(d_pet_dir, pet_dir)

                # e
                e_dir, _ = make_conv_dir(output_dir, name, e)
                
                # f
                f_dir, _ = make_conv_dir(output_dir, name, f)

                # gt
                gt_dirs = make_gt_dirs([a_dir, b_dir, c_dir, d_dir, e_dir, f_dir])
                gt_conv(input_path, gt_dirs, name, a_path)
        except:
            raise SystemError(f"{name} failed")
    make_dict(names, output_dir)

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
        
    pet_diffs = get_differences(pet_vals_dict)
    mask_diffs = get_differences(mask_vals_dict)

    get_hist(pet_diffs, case_dir)
    get_mask_matrix(mask_diffs, case_dir)
    get_subtracted_plots(pet_img_dict, case_dir)
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
        # print(entry)
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
    final_df.to_csv(os.path.join(output_dir, result_fi_name), index=False)
    
def make_pox_blot(df_conv_types, df_maes, output_dir):
    """
    Provide a list of the conversion types and all
    the corresponding mean absolute errors, provide 
    the boxplot for dicom2nifti, dcm2niix, dcmstack
    """
    conv_names = []
    data = []

    for key in box_plot_keys:
        conv_name = titles_dict[key]
        conv_names.append(conv_name)
        conv_data = []
        for ix, conv_type in enumerate(df_conv_types):
            if (conv_type==conv_name):
                conv_data.append(df_maes[ix])
        data.append(conv_data)

    plt.boxplot(data)
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.xticks([1,2,3], conv_names)
    plt.title("Mean Absolute Error Relative to SimpleITK")
    plt.xlabel("Conversion Method")
    plt.ylabel("Mean Absolute Errors (SUV)")
    plt.savefig(os.path.join(output_dir, box_plot_fi_name))
    
def make_mae_table(df_conv_types, df_maes, output_dir):
    """
    Compute all the mean average values and store them in .csv file
    """
    entries = []
    for key in keys:
        conv_name = titles_dict[key]
        conv_maes = []
        for ix, conv_type in enumerate(df_conv_types):
            if (conv_name==conv_type):
                mae = df_maes[ix]
                conv_maes.append(mae)
        avg_conv_maes = sum(conv_maes)/len(conv_maes)
        entry = [conv_name, avg_conv_maes]
        entries.append(entry)
    df = pd.DataFrame(entries, columns = mae_table_columns)
    df.to_csv(os.path.join(output_dir, mae_table_fi_name))
    
def get_mae_table_plot(output_dir):
     """ 
     Provide the box plots and table to
     illustrate the mean absolute errors
     """   
     result_fi_path = os.path.join(output_dir, result_fi_name)
     results = pd.read_csv(result_fi_path)
     df_conv_types = results[conv_type_csv_index]
     df_maes = results[mae_csv_index]
     make_pox_blot(df_conv_types, df_maes, output_dir)
     make_mae_table(df_conv_types, df_maes, output_dir)