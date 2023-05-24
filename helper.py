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
import matplotlib.pyplot as plt
import math
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")


# Constants
pet_dir_name = 'PET'
gt_dir_name = 'GT'
fi_ext = '.nii.gz'
sitk_path = './sitk.py'
ai4elife_dir_name = 'pet'
pred_dir_name = 'predicted_data'
dict_name = 'cases'
dict_ext = '.txt'
pet_end = 'pet.nii'
mask_end = 'predicted.nii'
fontsize = 24

keys = ['a', 'b', 'c', 'd', 'e', 'f']
a,b,c,d,e,f = keys
titles_dict = {'a' : 'dicom2nifti',
               'b' : 'dcm2niix',
               'c' : 'dcmstack',
               'd' : 'sitk',
               'e' : 'lifex',
               'f' : 'slicer'}

cut_dict = {0:'Sagittal', 1: 'Coronal'}

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


def file_conversion(input_dir, output_dir):
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
                suv_factor, Rescale_Slope, Rescale_Intercept = get_suv(pet_dir)

                #a
                a_dir = make_conv_dir(output_dir, name, a)
                a_path=a_conv(a_dir, pet_dir)
                scale_nifti(a_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                #b 
                b_dir = make_conv_dir(output_dir, name, b)
                b_path=b_conv(b_dir, pet_dir)
                scale_nifti(b_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                #c
                c_dir = make_conv_dir(output_dir, name, c)
                c_path=c_conv(c_dir, pet_dir)
                scale_nifti(c_path, suv_factor, Rescale_Slope, Rescale_Intercept)

                # d
                d_dir = make_conv_dir(output_dir, name, d)
                d_conv(d_dir, pet_dir)

                # e
                _ = make_conv_dir(output_dir, name, e)
                
                #f
                _ = make_conv_dir(output_dir, name, f)
        except:
            raise SystemError(f"{name} failed")
    make_dict(names, output_dir)

def read_dict(temp_dir):
    """
    Given the path to the directory holding our dictionary, provide a 
    list of all the cases
    """
    dict_path = os.path.join(temp_dir, dict_name+dict_ext)

    file = open(dict_path,"r")
    cases = file.readlines()

    return cases

def get_dirs(temp_dir):
    """
    Given the temporary directory path, provide a list of all
    the directory names
    """
    pred_dir = os.path.join(temp_dir, pred_dir_name)
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
def get_case_data(dirs, case, temp_dir):
    """
    Run through all the directories and get the relevant images for our case
    """
    case_data = []

    for dir in dirs:
        if dir.startswith(case):
            conv_type = dir.replace(case+"_", "")
            path = os.path.join(temp_dir, pred_dir_name, dir)
            for fi in os.listdir(path):
                fi_path = os.path.join(path, fi)
                if fi.endswith(pet_end):
                    pet_img = nib.load(fi_path)
                else:
                    mask_img = nib.load(fi_path)
            case_data.append([conv_type, pet_img, mask_img])
    return case_data

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

def get_cut_differences(dict, cut):
    """
    Given a specific cut: {1:coronal, 0:sagittal}, provide all
    of the subtracted images. The dictionary will have the images
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

    vals = np.array(cut_subs).reshape(no_rows, no_cols, 128, 256, 1)
    titles = get_titles()
    titles = np.array(titles).reshape(no_rows, no_cols)

    fig, axs = plt.subplots(no_rows,no_cols, figsize=(25,10), sharey=True, sharex=True, layout='constrained')
    fig.suptitle('Subtracted Plots', y=1, fontsize=24)

    for m in np.arange(no_rows):
        for n in np.arange(no_cols):
            axs[m][n].imshow(vals[m][n], cmap=cmap)
            axs[m][n].set_title(titles[m][n], size=18)
            axs[m][n].axis('off')
    fig.colorbar(im, ax=axs[:, n], location='right')
    plt.savefig(os.path.join(case_dir, f'{cut_dict[cut]} Subtracted Plots.png'))
    plt.clf()


def get_subtracted_plots(pet_img_dict, case_dir):
    """
    Make the subtracted plots for the given comparisons
    """
    cuts = [0,1]

    for cut in cuts:
        cut_subs = get_cut_differences(pet_img_dict, cut)
        get_sub_img(cut_subs, case_dir, cut)



def get_results(case_dir, case_data):
    """
    Given all the data, provide a confusion matrix of the mask values,
    histogram of the pet values absolute differences and subtracted images
    """
    pet_vals_dict = {}
    pet_img_dict = {}
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
        
    pet_diffs = get_differences(pet_vals_dict)
    mask_diffs = get_differences(mask_vals_dict)

    get_hist(pet_diffs, case_dir)
    get_mask_matrix(mask_diffs, case_dir)
    get_subtracted_plots(pet_img_dict, case_dir)
