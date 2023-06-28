# rt_struct_conversion
The purpose of this project is to understand the effects of different rt_struct conversion methods in addition to our dicom series conversion methods. As we know, the ground truth files are primarily provided in two formats: as a binary label map or as an rt_struct. For our purposes, we want all the ground truths to be provided as binary label maps. There are multiple conversion methods for this but we will be using our very own `rt_utils` variant, `dcmrtstruct2nii` and the built-in capabilities of LifeX and 3D Slicer.  

**High Level Methodology**: Convert our DICOM series PET files into a single NIfTI file using the 6 mentioned techniques (4 of which the code accomplishes, the other 2 are done manually) and repeat 4 times with each iteration changing the rt_struct conversion method. To simulate the effects on training, we send the data through `ai4elife`. We compare the raw predicted PET values and compare the predicted masks with the ground truth masks. 

# Methodology
The first step is to create the NIfTI files with the 4 mentioned methods. This will require the LifeX and 3D Slicer files to already have been manually converted to NIfTI. The required directory structure for the input directory for `create_nifti_files.py` is provided as follows
```
|-- main_dir                                                   <-- The main directory

|      |-- data_dir                                            <-- Directory containing all the relevant
                                                                   inputted dicom series files
|      |      |-- case_1                                       <-- Individual Folder with Unique ID
|      |      |      |-- PET                                   <-- The pet folder with .dcm files
                           |-- *.dcm                           <-- PET Image in .dcm format
                           |-- *.dcm                           <-- PET Image in .dcm format
                           .
                           .
                           |-- *.dcm                           <-- PET Image in .dcm format
|      |           |-- GT                                      <-- The ground truth folder with a .dcm file 
                           |-- *.dcm                           <-- GET Image in .dcm format (one file)
|      |      |-- case_2                                       <-- Individual Folder with Unique ID
|      |      |      |-- PET                                   <-- The pet folder with .dcm files
                           |-- *.dcm                           <-- PET Image in .dcm format
                           |-- *.dcm                           <-- PET Image in .dcm format
                           .
                           .
                           .
                           |-- *.dcm                           <-- PET Image in .dcm format
|      |      |      |-- GT                                    <-- The ground truth folder with a .dcm file 
                           |-- *.dcm                           <-- GET Image in .dcm format (one file)
              .
              .
              .
|      |      |-- case_n                                       <-- Individual Folder with Unique ID
|      |      |      |-- PET                                   <-- The pet folder with .dcm files
                           |-- *.dcm                           <-- PET Image in .dcm format
                           |-- *.dcm                           <-- PET Image in .dcm format
                           .
                           .
                           .
                           |-- *.dcm                           <-- PET Image in .dcm format
|      |      |      |-- GT                                    <-- The ground truth folder with a .dcm file 
                           |-- *.dcm                           <-- GET Image in .dcm format (one file)
                           
|      |-- lifex_slicer_dir                                    <-- Directory containing all the lifex and slicer
                                                                   NIfTI files (manually created)
|      |      |-- case_1_lifex                                 <-- Folder for case_1 lifex file
|      |      |      |-- PET                                   <-- The pet folder with nifti file
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format
|      |      |-- case_1_slicer                                <-- Folder for case_1 slicer file
|      |      |      |-- PET                                   <-- The pet folder with nifti file
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format   
              .
              .
              .
|      |      |-- case_n_lifex                                 <-- Folder for case_n lifex file
|      |      |      |-- PET                                   <-- The pet folder with nifti file
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format
|      |      |-- case_n_slicer                                <-- Folder for case_n slicer file
|      |      |      |-- PET                                   <-- The pet folder with nifti file
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format              
```
With this directory structure, run the following command (using the environment specified by `environment.yml`:
```
python create_nifti_files.py -m <path\to\main_dir>
```
After running this script with the appropriate environment, the `main_dir` will have the following structure (where `gt_conv_1`, `gt_conv_2`, `gt_conv_3`, `gt_conv_4` all have the same structure but only `gt_conv_4` for conciseness):
```
|-- main_dir                                                          <-- The main directory

|      |-- data_dir                                                   <-- Directory containing all the relevant
                                                                          inputted dicom series files
|      |-- lifex_slicer_dir                                           <-- Directory containing all the lifex and slicer
                                                                          NIfTI files (manually created)
|      |-- gt_conv_1                                                  <-- Directory with gt_conv_1 data/results
|      |-- gt_conv_2                                                  <-- Directory with gt_conv_2 data/results
|      |-- gt_conv_3                                                  <-- Directory with gt_conv_3 data/results
|      |-- gt_conv_4                                                  <-- Directory with gt_conv_4 data/results
|      |      |-- temp_dir
|      |      |      |-- case_1_convmethod_1                          <-- case_1 convmethod_1 folder
|      |      |      |      |-- gt                                    <-- The ground truth folder
                                  |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |      |-- pet                                   <-- The pet folder
                                  |-- *.nii.gz                        <-- PET Image in .nii.gz format
|      |      |      |-- case_1_convmethod_2                          <-- case_1 convmethod_2 folder
|      |      |      |      |-- gt                                    <-- The ground truth folder
                                  |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |      |-- pet                                   <-- The pet folder
                                  |-- *.nii.gz                        <-- PET Image in .nii.gz format
                     . 
                     .
                     .
|      |      |      |-- case_1_convmethod_6                          <-- case_1 convmethod_6 folder
|      |      |      |      |-- gt                                    <-- The ground truth folder
                                  |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |      |-- pet                                   <-- The pet folder
                                  |-- *.nii.gz                        <-- PET Image in .nii.gz format        
                     . 
                     .
                     .
|      |      |      |-- case_n_convmethod_6                          <-- case_n convmethod_6 folder
|      |      |      |      |-- gt                                    <-- The ground truth folder
                                  |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |      |-- pet                                   <-- The pet folder
                                  |-- *.nii.gz                        <-- PET Image in .nii.gz format
```

Now the `apply_ai4elife.py` script is run. Note that this should specify be run with the ai4elife environment (provided by the `ai4elife_environment.yml`). Its corresponding github repository must also be downloaded on the device as well. This is run with the following code where `ai4elife` is the downloaded git repository.
```
python apply_ai4elife.py -m </path/to/main_dir> -a <path\to\ai4elife\dir>
```
The main directory will then have the following structure:

```
|-- main_dir                                                          <-- The main directory

|      |-- data_dir                                                   <-- Directory containing all the relevant
                                                                          inputted dicom series files
|      |-- lifex_slicer_dir                                           <-- Directory containing all the lifex and slicer
                                                                          NIfTI files (manually created)
|      |-- gt_conv_1                                                  <-- Directory with gt_conv_1 data/results
|      |-- gt_conv_2                                                  <-- Directory with gt_conv_2 data/results
|      |-- gt_conv_3                                                  <-- Directory with gt_conv_3 data/results
|      |-- gt_conv_4                                                  <-- Directory with gt_conv_4 data/results
|      |      |-- temp_dir
|      |      |-- ai_dir                                              <-- Directory with the ai4elife output  
|      |      |      |-- data_default_3d_dir_                         <-- Temporary Directory
|      |      |      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                                  |-- ground_truth.nii                <-- ground truth file in .nii format
                                  |-- pet.nii                         <-- pet file in .nii format
|      |      |      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                                  |-- ground_truth.nii                <-- ground truth file in .nii format
                                  |-- pet.nii                         <-- pet file in .nii format 
                            . 
                            . 
                            . 
|      |      |      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                                  |-- ground_truth.nii                <-- ground truth file in .nii format
                                  |-- pet.nii                         <-- pet file in .nii format
|      |      |      |-- data_default_MIP_dir                         <-- Temporary Directory
|      |      |      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                                  |-- ground_truth_coronal.nii        <-- ground truth coronal file in .nii format
                                  |-- ground_truth_sagittal.nii       <-- ground truth sagittal file in .nii format
                                  |-- pet_coronal.nii                 <-- pet coronal file in .nii format
                                  |-- pet_sagittal.nii                <-- pet sagittal file in .nii format
|      |      |      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                                  |-- ground_truth_coronal.nii        <-- ground truth coronal file in .nii format
                                  |-- ground_truth_sagittal.nii       <-- ground truth sagittal file in .nii format
                                  |-- pet_coronal.nii                 <-- pet coronal file in .nii format
                                  |-- pet_sagittal.nii                <-- pet sagittal file in .nii format
                            . 
                            . 
                            . 
|      |      |      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                                  |-- ground_truth_coronal.nii        <-- ground truth coronal file in .nii format
                                  |-- ground_truth_sagittal.nii       <-- ground truth sagittal file in .nii format
                                  |-- pet_coronal.nii                 <-- pet coronal file in .nii format
                                  |-- pet_sagittal.nii                <-- pet sagittal file in .nii format

|      |      |      |-- predicted
|      |      |      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                                  |-- ""_ground_truth.nii             <-- ground truth coronal file in .nii format
                                  |-- ""_pet.nii                      <-- pet file in .nii format
                                  |-- ""_predicted.nii                <-- predicted masks in .nii format
|      |      |      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                                  |-- ""_ground_truth.nii             <-- ground truth coronal file in .nii format
                                  |-- ""_pet.nii                      <-- pet file in .nii format
                                  |-- ""_predicted.nii                <-- predicted masks in .nii format
                            . 
                            . 
                            . 
|      |      |      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                                   |-- ""_ground_truth.nii             <-- ground truth coronal file in .nii format
                                   |-- ""_pet.nii                      <-- pet file in .nii format
                                   |-- ""_predicted.nii                <-- predicted masks in .nii format
|      |      |      |-- cases.txt                                    <-- list of all cases in .txt format
|      |      |      |-- surrogate_ground_truth.csv                   <-- TMTV and relevant info for ground truths as .csv
|      |      |      |-- surrogate_predicted.csv                      <-- TMTV and relevant info for predicted PET as .csv
```
The final step is to compile all the results. Within each ground truth conversion method, for each case, a single .csv file is created containing the conversion types, dice score and mean absolute errors. For each ground truth conversion method, these .csv files are combined together with the ai4elife surrogate .csv files to all the results for that one conversion method. These are all temporary .csv files.

These .csv files are combined to provide the `combined_results.csv` file. `gt_comparison_results.csv` average the values over all the cases for each ground truth conversion method. And five box plots (two for dice scores and three for TMTV) are provided to visualize this data. The following command provides all of these results. 
```
python compare_results.py -m <path\to\main_dir>
```
The final structure of the `main_dir` will be provided as follows:
```
|-- main_dir                                                          <-- The main director

|      |-- data_dir                                                   <-- Directory containing all the relevant
                                                                          inputted dicom series files
|      |-- lifex_slicer_dir                                           <-- Directory containing all the lifex and slicer
                                                                          NIfTI files (manually created)
|      |-- gt_conv_1                                                  <-- Directory with gt_conv_1 data/results
|      |-- gt_conv_2                                                  <-- Directory with gt_conv_2 data/results
|      |-- gt_conv_3                                                  <-- Directory with gt_conv_3 data/results
|      |-- gt_conv_4                                                  <-- Directory with gt_conv_4 data/results
|      |      |-- temp_dir
|      |      |-- ai_dir                                              <-- Directory with the ai4elife output 
|      |      |-- results_dir                                         <-- Directory with the results
|      |      |      |-- case_1                                       <-- case_1 results directory
                           |-- case_1.csv                             <-- case_1 results in .csv format
                     .
                     .
                     .
|      |      |      |-- case_n                                       <-- case_n results directory
                           |-- case_n.csv                             <-- case_n results in .csv format
                     |-- results.csv                                  <-- combined results for gt_conv_4 in .csv format
                     
|      |-- combined_results.csv                                       <-- combined results for all gt_conv in .csv format
|      |-- gt_comparison_results.csv                                  <-- averaged results for each gt_conv in .csv format
|      |-- Coronal_Dice_box_plot.png    
|      |-- Sagittal_Dice_box_plot.png 
|      |-- Coronal_TMTV_box_plot.png 
|      |-- Sagittal_TMTV_box_plot.png  
|      |-- Total_TMTV_box_plot.png 
```
