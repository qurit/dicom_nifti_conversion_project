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
        '''))
argParser.add_argument("-i", "--input_dir", help="path to dir with DICOM series folders", type=str, required=True)
argParser.add_argument("-o", "--output_dir", help="path to dir where NIfTI files will be saved", type=str, required=True)
args = argParser.parse_args()

sys.stdout.write("\n"+f"-"*100)

# For easy reference
input_dir = args.input_dir
output_dir = args.output_dir

# Check that the provided input directory is suitable
check_input_dir(input_dir)
sys.stdout.write("\n"+f"-"*100+ "\n")

# Create output directory if necessary
create_output_dir(output_dir)

# Do the file conversions
file_conversion(input_dir, output_dir)
sys.stdout.write("\n"+f"-"*100+ "\n")