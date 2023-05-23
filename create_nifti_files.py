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
argParser.add_argument("-a", "--a", help='1 if dicom2nifti is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-b", "--b", help='1 if dcm2niix is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-c", "--c", help='1 if dcmstack is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-d", "--d", help='1 if sitk is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-e", "--e", help='1 if lifex is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-f", "--f", help='1 if slicer is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
args = argParser.parse_args()

sys.stdout.write("\n"+f"-"*100)

# For easy reference
input_dir = args.input_dir
output_dir = args.output_dir
do_a, do_b, do_c, do_d, do_e, do_f=args.a, args.b, args.c, args.d, args.e, args.f 

# Check that the provided input directory is suitable
check_input_dir(input_dir)
sys.stdout.write("\n"+f"-"*100+ "\n")

# Create output directory if necessary
create_output_dir(output_dir)

# Do the file conversions
file_conversion(input_dir, output_dir, do_a, do_b, do_c, do_d, do_e, do_f)
sys.stdout.write("\n"+f"-"*100+ "\n")