# From code sharing to sharing of implementations: Advancing reproducible AI development for medical imaging through federated testing
This repository covers a part of the conversion techniques we used to analyze the effects of different DICOM to NIfTI file conversion on Medical Imaging Training. 

# Main Directories
This project consists of two main directories: `series_conversion` and `rt_struct_conversion`. With the `series_conversion` directory, we only focus on the differences between the dicom series and NIfTI conversion using six different methods. In the process of this experiment, we convert the ground truth dicom files based on whether they are in a binary map or rt-struct format. However, we use the same single conversion method for each of these. With `rt_struct_conversion` on the other hand, we are considering the differences between the dicome series to NIfTI file conversion (using the same six methods) and the rt_struct dicom file to NIfTI file (using four different methods).

Each of these directories will contain the relevant instructions for running multiple scripts and switching between the environments. 
