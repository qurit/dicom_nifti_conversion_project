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
