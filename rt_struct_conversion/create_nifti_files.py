#!/usr/bin/env python
from helper import *

argParser = argparse.ArgumentParser(
    prog='PROG',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        Creation of NIfTI Files
        --------------------------------
            This code will create the NIfTI files for the 
            dicom2nifti, dcm2niix and dcmstack conversion
            methods. It will provide these in directories
            as required by the ai4elife program. It will also
            create the directories for lifex and 3D-slicer but
            these NIfTI files must be manually created. Finally
            it will convert the ground truth files and copy them
            into the appropriate directories for ai4elife. 
        --------------------------------
            This code will use the following convention for 
            referring to the different conversion methods.
            -a : dicom2nifti
            -b : dcm2niix
            -c : dcmstack
            -d : sitk
            -e : lifex
            -f : slicer
        --------------------------------
            For the different gt rt_struct conversion methods, the program
            will refer to them as:
            -'x' : rt_utils
            -'y' : pyradise
            -'z' : dcmrtstruct2nii
            -'u' : lifex
            -'v' : slicer
        --------------------------------
            The prompted main directory is the directory
            where the following directories are/will be saved:
            -data_dir
            -lifex_slicer_dir
            -dcmrtstruct2nii
            -lifex
            -rt_utils
            -slicer
        '''))
argParser.add_argument("-m", "--main_dir", help="path to main directory", type=str, required=True)

args = argParser.parse_args()

main_dir = args.main_dir
data_dir = os.path.join(main_dir, data_dir_name)
lifex_slicer_dir = os.path.join(main_dir, lifex_slicer_dir_name)

key_dirs, cases = create_fis(data_dir, main_dir)
process_lifex_slicer_fis(key_dirs, lifex_slicer_dir, cases)
coordinate_dir(main_dir)

