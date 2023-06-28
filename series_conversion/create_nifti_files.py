#!/usr/bin/env python3

from helper import *

# Arguments to be passed
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
            The prompted main directory is the directory
            where the following directories are/will be saved:
            -data_dir
            -temp_dir
            -ai_dir
            -results_dir
            -lifex_slicer_dir
        '''))
argParser.add_argument("-m", "--main_dir", help="path to dir where all data/results will be saved", type=str, required=True)
args = argParser.parse_args()

sys.stdout.write("\n"+f"-"*100)

# For easy reference
main_dir = args.main_dir
data_dir = os.path.join(main_dir, data_dir_name)
lifex_slicer_dir = os.path.join(main_dir, lifex_slicer_dir_name)

# Check that the provided input directory is suitable
check_input_dir(data_dir)
sys.stdout.write("\n"+f"-"*100+ "\n")

# Create output directory if necessary
temp_dir = create_main_dir(main_dir)

# Do the file conversions
names = file_conversion(data_dir, temp_dir)
sys.stdout.write("\n"+f"-"*100+ "\n")

# Move over the LIFEx and Slicer files
process_lifex_slicer_fis(temp_dir, lifex_slicer_dir, cases=names)
coordinate_fis(temp_dir)