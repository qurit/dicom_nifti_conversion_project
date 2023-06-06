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
        '''))
argParser.add_argument("-i", "--input_dir", help="path to dir with DICOM series folders", type=str, required=True)
argParser.add_argument("-o", "--output_dir", help="path to dir where all directories will be saved (input for future scripts)", type=str, required=True)
argParser.add_argument("-ls", "--lifex_slicer_dir", help="path to dir where manually created lifex and slicer NIfTI files are stored", type=str, required=True)

args = argParser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
lifex_slicer_dir = args.lifex_slicer_dir

key_dirs, cases = create_fis(input_dir, output_dir)
process_lifex_slicer_fis(key_dirs, lifex_slicer_dir, cases)
coordinate_dir(output_dir)

