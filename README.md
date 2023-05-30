# dicom_nifti_project
Analyzing the effects of different DICOM to NIfTI file conversion on Medical Imaging Training. 

# Introduction

The purpose of this repository is to explore the effects of different DICOM to NIfTI file conversion techniques on Medical imaging AI training models. Because these two file formats are used extensively in this area of research, it is pivotal to understand the effects of changing between them. In this repository, we will explicitly be dealing with `dicom2nifti`, `dcm2niix`, `dcmstack`, a script using `SimpleITK` in addition to built-in functions in LIFEx and 3D Slicer Applications. 

**High Level Methodology**: Convert our DICOM series PET files into a single NIfTI file using the 6 mentioned techniques (4 of which the code accomplishes, the other 2 are done manually). To simulate the effects on training, we send the data through `ai4elife`. We compare the raw predicted PET values and compare the predicted masks with the ground truth masks. 

# METHODOLOGY
The first step is to create the NIfTI files with the 4 mentioned methods (not LIFEx and 3D slicer as these must be manually done). 
The required directory structure for the input directory for `create_nifti_files.py` is provided as follows
```
|-- input folder                                        <-- The main folder of all input PET and GT files

|      |-- parent folder (case_1)                       <-- Individual Folder with Unique ID
|           |-- PET                                     <-- The pet folder with .dcm files
                 | -- *.dcm                             <-- PET Image in .dcm format
                 | -- *.dcm                             <-- PET Image in .dcm format
                 .
                 .
                 .
                 | -- *.dcm                             <-- PET Image in .dcm format
|           |-- GT                                      <-- The ground truth folder with a .dcm file 
                 | -- *.dcm                             <-- GET Image in .dcm format (one file)
|      |-- parent folder (case_2)                       <-- Individual Folder with Unique ID
|           |-- PET                                     <-- The pet folder with .dcm files
                 | -- *.dcm                             <-- PET Image in .dcm format
                 | -- *.dcm                             <-- PET Image in .dcm format
                 .
                 .
                 .
                 | -- *.dcm                             <-- PET Image in .dcm format
|           |-- GT                                      <-- The ground truth folder with a .dcm file 
                 | -- *.dcm                             <-- GET Image in .dcm format (one file)
|           .
|           .
|           .
|      |-- parent folder (case_n)                       <-- Individual Folder with Unique ID
|           |-- PET                                     <-- The pet folder with .dcm files
                 | -- *.dcm                             <-- PET Image in .dcm format
                 | -- *.dcm                             <-- PET Image in .dcm format
                 .
                 .
                 .
                 | -- *.dcm                             <-- PET Image in .dcm format
|           |-- GT                                      <-- The ground truth folder with a .dcm file 
                 | -- *.dcm                             <-- GET Image in .dcm format (one file)
```
With this directory structure, run the following command where temp_folder is the output:
```
python create_nifti_files.py -i <path\to\input\dir> -o <path\to\temp_folder>
```
This will provide the following directory structure:
```
|-- temp_folder                                         <-- Output folder of create_nifti_files.py,
                                                            Input folder of apply_ai4elife.py
|      |-- case_1_convmethod_1                          <-- case_1 convmethod_1 folder
|           |-- gt                                      <-- The ground truth folder
                 | -- *.nii.gz                          <-- GT Image in .nii.gz format
|           |-- pt                                      <-- The pet folder
                 | -- *.nii.gz                          <-- PET Image in .nii.gz format
|      |-- case_1_convmethod_2                          <-- case_1 convmethod_2 folder
|           |-- gt                                      <-- The ground truth folder
                 | -- *.nii.gz                          <-- GT Image in .nii.gz format
|           |-- pt                                      <-- The pet folder
                 | -- *.nii.gz                          <-- PET Image in .nii.gz format
            .
            .
            .
|      |-- case_1_convmethod_6                          <-- case_1 convmethod_6 folder
|           |-- gt                                      <-- The ground truth folder
                 | -- *.nii.gz                          <-- GT Image in .nii.gz format
|           |-- pt                                      <-- The pet folder
                 | -- *.nii.gz                          <-- PET Image in .nii.gz format           
|           .
|           .
|           .
|      |-- case_n_convmethod_6                          <-- case_1 convmethod_6 folder
|           |-- gt                                      <-- The ground truth folder
                 | -- *.nii.gz                          <-- GT Image in .nii.gz format
|           |-- pt                                      <-- The pet folder
                 | -- *.nii.gz                          <-- PET Image in .nii.gz format 
```

After the creation of the NIfTI files (using the `dicom2nifti`, `dcm2niix`, `dcmstack` and `SimpleITK` methods) the user should add the manually generated LIFEx and 3D Slicer NIfTI files to their respective directories in thet above structure. Afterwards, to coordinate all of these conversion methods, the `coordinate.py` script is ran using our usual environment. The following command is run for this:
```
python coordinate.py -i </path/to/temp_folder>
```
This will leave the directory in the exact same structure but will have loaded and saved all the NIfTI files using Nibabel (to elimniate biases) so that after loading certain cases, we can apply the necessary rotations. Afterwards, the `apply_ai4elife.py` script is ran. Note that this should specify be run with the ai4elife environment. Its corresponding github repository must also be downloaded on the device as well. This is run with the following code where `ai4elife_folder` is the ouput directory. 
```
python apply_ai4elife.py -i </path/to/temp_folder> -o <path\to\ai4elife_folder> -a <path\to\ai4elife\dir>
```

