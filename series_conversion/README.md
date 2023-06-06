# series_conversion
To purpose of this directory is to consider the effects on training of different dicom series to NIfTI conversion techniques. The 6 methods that will be used are `dicom2nifti`, `dcm2niix`, `dcmstack`, a script using `SimpleITK` in addition to built-in functions in LIFEx and 3D Slicer Applications. In order to simulate training, we will be using `ai4elife` and for our ground truths, we will be using our `rt_utils` dicom rt-struct to single NIfTI file conversion or our `dcm2nii` function (when the ground truth is presented as a binary label map). 

**High Level Methodology**: Convert our DICOM series PET files into a single NIfTI file using the 6 mentioned techniques (4 of which the code accomplishes, the other 2 are done manually). To simulate the effects on training, we send the data through `ai4elife`. We compare the raw predicted PET values and compare the predicted masks with the ground truth masks. 

# METHODOLOGY
The first step is to create the NIfTI files with the 4 mentioned methods. This will require the LifeX and 3D Slicer files to already have been manually converted to NIfTI. The required directory structure for the input directory for `create_nifti_files.py` is provided as follows
```
|-- main_dir                                                   <-- The main directory of all input PET and GT files

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
With this directory structure, run the following command where temp_folder is the output (using the environment specified by `environment.yml`:
```
python create_nifti_files.py -m <path\to\main_dir>
```
After running this script with the appropriate environment, the `main_dir` will have the following structure:
```
|-- main_dir                                                   <-- The main directory of all input PET and GT files

|      |-- data_dir                                            <-- Directory containing all the relevant
                                                                   inputted dicom series files
|      |-- lifex_slicer_dir                                    <-- Directory containing all the lifex and slicer
                                                                   NIfTI files (manually created)
|      |-- temp_dir                                            <-- Directory with the converted and
                                                                   coordinated NIfTI files (sent to ai4elife)                                  
|      |      |-- case_1_convmethod_1                          <-- case_1 convmethod_1 folder
|      |      |      |-- gt                                    <-- The ground truth folder
                           |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |-- pet                                   <-- The pet folder
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format
|      |      |-- case_1_convmethod_2                          <-- case_1 convmethod_2 folder
|      |      |      |-- gt                                    <-- The ground truth folder
                           |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |-- pet                                   <-- The pet folder
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format
              . 
              .
              .
|      |      |-- case_1_convmethod_6                          <-- case_1 convmethod_6 folder
|      |      |      |-- gt                                    <-- The ground truth folder
                           |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |-- pet                                   <-- The pet folder
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format        
              . 
              .
              .
|      |      |-- case_n_convmethod_6                          <-- case_n convmethod_6 folder
|      |      |      |-- gt                                    <-- The ground truth folder
                           |-- *.nii.gz                        <-- GT Image in .nii.gz format
|      |      |      |-- pet                                   <-- The pet folder
                           |-- *.nii.gz                        <-- PET Image in .nii.gz format
```

Now the `apply_ai4elife.py` script is run. Note that this should specify be run with the ai4elife environment (provided by the `ai4elife_environment.yml`). Its corresponding github repository must also be downloaded on the device as well. This is run with the following code where `ai4elife` is the downloaded git repository.
```
python apply_ai4elife.py -m </path/to/main_dir> -a <path\to\ai4elife\dir>
```
The main directory will then have the following structure:

```
|-- main_dir                                                   <-- The main directory of all input PET and GT files

|      |-- data_dir                                            <-- Directory containing all the relevant
                                                                   inputted dicom series files
|      |-- lifex_slicer_dir                                    <-- Directory containing all the lifex and slicer
                                                                   NIfTI files (manually created)
|      |-- temp_dir                                            <-- Directory with the converted and
                                                                   coordinated NIfTI files (sent to ai4elife)  
|      |-- ai_dir                                              <-- Directory with the ai4elife output  
|      |      |-- data_default_3d_dir_                         <-- Temporary Directory
|      |      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                           |-- ground_truth.nii                <-- ground truth file in .nii format
                           |-- pet.nii                         <-- pet file in .nii format
|      |      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                           |-- ground_truth.nii                <-- ground truth file in .nii format
                           |-- pet.nii                         <-- pet file in .nii format 
                     . 
                     . 
                     . 
|      |      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                           |-- ground_truth.nii                <-- ground truth file in .nii format
                           |-- pet.nii                         <-- pet file in .nii format
|      |      |-- data_default_MIP_dir                         <-- Temporary Directory
|      |      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                           |-- ground_truth_coronal.nii        <-- ground truth coronal file in .nii format
                           |-- ground_truth_sagittal.nii       <-- ground truth sagittal file in .nii format
                           |-- pet_coronal.nii                 <-- pet coronal file in .nii format
                           |-- pet_sagittal.nii                <-- pet sagittal file in .nii format
|      |      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                           |-- ground_truth_coronal.nii        <-- ground truth coronal file in .nii format
                           |-- ground_truth_sagittal.nii       <-- ground truth sagittal file in .nii format
                           |-- pet_coronal.nii                 <-- pet coronal file in .nii format
                           |-- pet_sagittal.nii                <-- pet sagittal file in .nii format
                     . 
                     . 
                     . 
|      |      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                           |-- ground_truth_coronal.nii        <-- ground truth coronal file in .nii format
                           |-- ground_truth_sagittal.nii       <-- ground truth sagittal file in .nii format
                           |-- pet_coronal.nii                 <-- pet coronal file in .nii format
                           |-- pet_sagittal.nii                <-- pet sagittal file in .nii format

|      |      |-- predicted
|      |      |      |-- case_1_convmethod_1                   <-- case_1 convmethod_1 folder
                           |-- ""_ground_truth.nii             <-- ground truth coronal file in .nii format
                           |-- ""_pet.nii                      <-- pet file in .nii format
                           |-- ""_predicted.nii                <-- predicted masks in .nii format
|      |      |      |-- case_1_convmethod_2                   <-- case_1 convmethod_2 folder
                           |-- ""_ground_truth.nii             <-- ground truth coronal file in .nii format
                           |-- ""_pet.nii                      <-- pet file in .nii format
                           |-- ""_predicted.nii                <-- predicted masks in .nii format
                     . 
                     . 
                     . 
|      |      |      |-- case_n_convmethod_6                   <-- case_n convmethod_6 folder
                           |-- ""_ground_truth.nii             <-- ground truth coronal file in .nii format
                           |-- ""_pet.nii                      <-- pet file in .nii format
                           |-- ""_predicted.nii                <-- predicted masks in .nii format
|      |      |-- cases.txt                                    <-- list of all cases in .txt format
|      |      |-- surrogate_ground_truth.csv                   <-- TMTV and relevant info for ground truths as .csv
|      |      |-- surrogate_predicted.csv                      <-- TMTV and relevant info for predicted PET as .csv
```
The final step is to compile all the results. For each individual case, there are four results file (in addition to a fifth temporary .csv file which will be discussed later on). The `histogram.png` plot is looking at the raw absolute value differences between the different methods for the sagittal and coronal PET images (at the same time). After iterating over both images and all dimensions, we calculate the absolute value difference and plot the histogram. The `Coronal_subtracted_plots.png` and the `Sagittal_subtracted_plots.png` display the absolute value difference between the different conversion methods as a plot. Regions with a high value indicate that the methods differed considerably there. Finally, `mask_vals.png` is a confusion matrix showing the number of predicted_masks indices (with values of either 0 or 1) that differed between the cases. 

To neatly compile the information from the surrogate files and our calculated dice scores and mean average errors, we provide a `results.csv` file with all of this information. In order to compare the results to our reference `SimpleITK` method, we also provide a `mae_results.csv` file and a boxplot showing the absolute errors between `dicom2nifti`, `dcm2niix` and `dcmstack` relative to `SimpleITK`. For the other methods (`lifex`, `slicer`), the differences were at least 8 orders of magnitude larger so these weren't plotted but they are provided in the `mae_results.csv`. The following command provides all of these results. 
```
python compare_results.py -m <path\to\main_dir>
```
The final structure of the `main_dir` will be provided as follows:
```
|-- main_dir                                                   <-- The main directory of all input PET and GT files

|      |-- data_dir                                            <-- Directory containing all the relevant
                                                                   inputted dicom series files
|      |-- lifex_slicer_dir                                    <-- Directory containing all the lifex and slicer
                                                                   NIfTI files (manually created)
|      |-- temp_dir                                            <-- Directory with the converted and
                                                                   coordinated NIfTI files (sent to ai4elife)  
|      |-- ai_dir                                              <-- Directory with the ai4elife output  
|      |-- results_dir                                         <-- Directory with all of the results
|      |      |-- case_1                                       <-- case_1 results folder
                    |-- case_1.csv                             <-- Temp case results file in .csv formay
                    |-- Coronal_subtracted_plots.png            
                    |-- Sagittal_subtracted_plots.png
                    |-- histogram.png
                    |-- mask_vals.png
|      |      |-- case_2                                       <-- case_2 results folder
                    |-- case_2.csv                             <-- Temp case results file in .csv formay
                    |-- Coronal_subtracted_plots.png            
                    |-- Sagittal_subtracted_plots.png
                    |-- histogram.png
                    |-- mask_vals.png
              .
              .
              .
|      |      |-- case_n                                       <-- case_n results folder
                    |-- case_n.csv                             <-- Temp case results file in .csv formay
                    |-- Coronal_subtracted_plots.png            
                    |-- Sagittal_subtracted_plots.png
                    |-- histogram.png
                    |-- mask_vals.png
                            
       |-- results.csv                                  <-- Final results file in .csv format
                                                            (Dice, MAE, TMTV)
       |-- mae_results.csv                              <-- Averaged MAE results in .csv format
       |-- box_plot.png                                 <-- MAE results as box plot (for dicom2nifti, dcm2niix, dcmstack)
```
# SAMPLE RESULTS
Directory `trial_data` contains sample predicted masks and all the results from two separates trials: `a_trial` and `b_trial`. For these two trials, the entire methodology was run in the same way twice from scratch. However, the LIFEx and 3D slicer files were the same between the two. In each individual trial's directory, there is `ai_data` which contains the masks and the TMTV values and `results` which contains all of the plots and final results .csv file. 
