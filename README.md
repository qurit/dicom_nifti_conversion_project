# dicom_nifti_project
Analyzing the effects of different DICOM to NIfTI file conversion on Medical Imaging Training. 

# Introduction

The purpose of this repository is to explore the effects of different DICOM to NIfTI file conversion techniques on Medical imaging AI training models. Because these two file formats are used extensively in this area of research, it is pivotal to understand the effects of changing between them. In this repository, we will explicitly be dealing with `dicom2nifti`, `dcm2niix`, `dcmstack`, `SimpleITK` in addition to LIFEx and 3D Slicer Applications. 

With a focus on PET values, we will be comparing the raw data and generated images by sending the data through `ai4elife`. 

# Directory Structure
This is the required directory structure for the input directory for `create_nifti_files.py`:
```
|-- input folder                                        <-- The main folder of all input PET and GT files

|      |-- parent folder (case_1)                       <-- Individual Folder with Unique ID
|           |-- PET                                     <-- The pet folder with .dcm files
                 | -- *.dcm                             <-- PET Image in .dcm format (multiple files)
|           |-- GT                                      <-- The ground truth folder with a .dcm file 
                 | -- *.dcm                             <-- GET Image in .dcm format (one file)
|      |-- parent folder (case_2)                       <-- Individual Folder with Unique ID
|           |-- PET                                     <-- The pet folder with .dcm files
                 | -- *.dcm                             <-- PET Image in .dcm format (multiple files)
|           |-- GT                                      <-- The ground truth folder with a .dcm file 
                 | -- *.dcm                             <-- GET Image in .dcm format (one file)
|           .
|           .
|           .
|      |-- parent folder (case_n)                       <-- Individual Folder with Unique ID
|           |-- PET                                     <-- The pet folder with .dcm files
                 | -- *.dcm                             <-- PET Image in .dcm format (multiple files)
|           |-- GT                                      <-- The ground truth folder with a .dcm file 
                 | -- *.dcm                             <-- GET Image in .dcm format (one file)
```

# Procedure
Firstly create all the relevant files (with conversions) with the following command:
```
python create_nifti_files.py -i <path\to\input\dir> -o <path\to\output\dir>
```
You can additionally specify which conversion methods to be used (by default) all of them are selected. Consult the following for an extensive list:

```
usage: PROG [-h] -i INPUT_DIR -o OUTPUT_DIR

Creation of NIfTI Files
--------------------------------
    This code will create the NIfTI files for the 
    dicom2nifti, dcm2niix and dcmstack conversion
    methods. It will provide these in directories
    as required by the ai4elife program. It will also
    create the directories for lifex and 3D-slicer but
    these NIfTI files must be manually created
--------------------------------
    This code will use the following convention for 
    referring to the different conversion methods.
    -a : dicom2nifti
    -b : dcm2niix
    -c : dcmstack
    -d : sitk
    -e : lifex
    -f : slicer

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        path to dir with DICOM series folders
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to dir where NIfTI files will be saved
```

After the creation of the NIfTI files (using the `dicom2nifti`, `dcm2niix`, `dcmstack` and `SimpleITK` methods) the user should add the manually generated LIFEx and 3D Slicer NIfTI files to their respective directories as well in the output directory. Afterwards, the `apply_ai4elife.py` script is ran. Note that this should specify be run with the ai4elife environment. Its corresponding github repository must also be downloaded on the device as well. This is run with the following code:
```
python apply_ai4elife.py -i <path\to\input\dir> -o <path\to\output\dir> -a <path\to\ai4elife\dir>
```
An explanation of the parameters can be found here:
```
usage: PROG [-h] -i INPUT_DIR -o OUTPUT_DIR -a AI_DIR

Creation of send NIfTI files through ai4elife

This must be run using the ai4elife specific environment

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        path to dir with patient nifti file directories
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to dir where processed NIfTI files will be saved
  -a AI_DIR, --ai_dir AI_DIR
                        path to dir with ai4elife (downloaded from github)
```
