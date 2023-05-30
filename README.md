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
This will leave the directory in the exact same structure but will have loaded and saved all the NIfTI files using Nibabel (to elimniate biases) so that after loading certain cases, we can apply the necessary rotations. Afterwards, the `apply_ai4elife.py` script is ran. Note that this should specify be run with the ai4elife environment. Its corresponding github repository must also be downloaded on the device as well. This is run with the following code where `ai4elife_folder` is the ouput directory and `ai4elife` is the downloaded directory
```
python apply_ai4elife.py -i </path/to/temp_folder> -o <path\to\ai4elife_folder> -a <path\to\ai4elife\dir>
```
The output of this directory is given as follows:
```
|-- ai4elife_folder                                     <-- Output folder of apply_ai4elife.py,
                                                            Input folder of compare_results.py
|      |-- data_default_3d_dir_                         <-- Temporary Directory
|      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                     | ground_truth.nii                 <-- ground truth file in .nii format
                     | pet.nii                          <-- pet file in .nii format
|      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                     | ground_truth.nii                 <-- ground truth file in .nii format
                     | pet.nii                          <-- pet file in .nii format  
              . 
              . 
              . 
|      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                     | ground_truth.nii                 <-- ground truth file in .nii format
                     | pet.nii                          <-- pet file in .nii format  
|      |-- data_default_MIP_dir                         <-- Temporary Directory
|      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                     | ground_truth_coronal.nii         <-- ground truth coronal file in .nii format
                     | ground_truth_sagittal.nii        <-- ground truth sagittal file in .nii format
                     | pet_coronal.nii                  <-- pet coronal file in .nii format
                     |pet_sagittal.nii                  <-- pet sagittal file in .nii format
|      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                     | ground_truth_coronal.nii         <-- ground truth coronal file in .nii format
                     | ground_truth_sagittal.nii        <-- ground truth sagittal file in .nii format
                     | pet_coronal.nii                  <-- pet coronal file in .nii format
                     |pet_sagittal.nii                  <-- pet sagittal file in .nii format 
              . 
              . 
              . 
|      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                     | ground_truth_coronal.nii         <-- ground truth coronal file in .nii format
                     | ground_truth_sagittal.nii        <-- ground truth sagittal file in .nii format
                     | pet_coronal.nii                  <-- pet coronal file in .nii format
                     |pet_sagittal.nii                  <-- pet sagittal file in .nii format

|      |-- predicted
|      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                     | ""_ground_truth.nii              <-- ground truth coronal file in .nii format
                     | ""_pet.nii                       <-- pet file in .nii format
                     | ""_predicted.nii                 <-- predicted masks in .nii format
|      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                     | ""_ground_truth.nii              <-- ground truth coronal file in .nii format
                     | ""_pet.nii                       <-- pet file in .nii format
                     | ""_predicted.nii                 <-- predicted masks in .nii format
              . 
              . 
              . 
|      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                     | ""_ground_truth.nii              <-- ground truth coronal file in .nii format
                     | ""_pet.nii                       <-- pet file in .nii format
                     | ""_predicted.nii                 <-- predicted masks in .nii format
       |-- cases.txt                                    <-- list of all cases in .txt format
       |-- surrogate_ground_truth.csv                   <-- TMTV and relevant info for ground truths as .csv
       |-- surrogate_predicted.csv                      <-- TMTV and relevant info for predicted PET as .csv
```
The final step is to compile all the results. For each individual case, there are four results file (in addition to a fifth temporary .csv file which will be discussed later on). The `histogram.png` plot is looking at the raw absolute value differences between the different methods for the sagittal and coronal PET images (at the same time). After iterating over both images and all dimensions, we calculate the absolute value difference and plot the histogram. The `Coronal_subtracted_plots.png` and the `Sagittal_subtracted_plots.png` display the absolute value difference between the different conversion methods as a plot. Regions with a high value indicate that the methods differed considerably there. Finally, `mask_vals.png` is a confusion matrix showing the number of predicted_masks indices (with values of either 0 or 1) that differed between the cases. 

To neatly compile the information from the surrogate files and our calculated dice scores and mean average errors, we provide a `results.csv` file with all of this information. To compile the results, the following command is run where `results_folder` is our output folder:
```
python compare_results.py -i <path\to\ai4elife_folder> -o <path\to\results_folder> -t <path\to\temp_folder>
```
This will provide the previously indicated results in the following structure:
```
|-- results_folder                                      <-- Output folder of compare_results.py,
                                                            Final results folder
|      |-- case_1                                       <-- case_1 results folder
            |-- case_1.csv                              <-- Temporary results file in .csv format
                                                            (Dice Score and Mean Absolute Error)
            |-- Coronal_subtracted_plots.png            
            |-- Sagittal_subtracted_plots.png
            |-- histogram.png
            |-- mask_vals.png
|      |-- case_2                                       <-- case_2 results folder
            |-- case_1.csv                              <-- Temporary results file in .csv format
                                                            (Dice Score and Mean Absolute Error)
            |-- Coronal_subtracted_plots.png            
            |-- Sagittal_subtracted_plots.png
            |-- histogram.png
            |-- mask_vals.png
       .
       .
       .
|      |-- case_n                                       <-- case_n results folder
            |-- case_1.csv                              <-- Temporary results file in .csv format
                                                            (Dice Score and Mean Absolute Error)
            |-- Coronal_subtracted_plots.png            
            |-- Sagittal_subtracted_plots.png
            |-- histogram.png
            |-- mask_vals.png
                            
       |-- results.csv                                  <-- Final results file in .csv format
                                                            (Dice, MAE, TMTV)
```
# SAMPLE RESULTS
Directory `a_trial` and `b_trial` are all the results from following this same procedure twice from scratch. However, both of these methods use the same manually generated lifex and slicer files. 
