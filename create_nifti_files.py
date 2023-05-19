#!/usr/bin/env python3

# Necessary Imports
import sys
import os
import argparse
import textwrap

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
argParser.add_argument("-a", "--a", help='1 if dicom2nifti is considered, 0 otherwise', type=int, required=False, default=True)
argParser.add_argument("-b", "--b", help='1 if dcm2niix is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-c", "--c", help='1 if dcmstack is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-d", "--d", help='1 if sitk is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-e", "--e", help='1 if lifex is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
argParser.add_argument("-f", "--f", help='1 if slicer is considered, 0 otherwise', type=int, choices = [0,1], required=False, default=True)
args = argParser.parse_args()

# Constants
pet_dir_name = 'PET'
gt_dir_name = 'GT'
fi_ext = '.nii.gz'
sitk_path = './sitk.py'
ai4elife_dir_name = 'pet'

a,b,c,d,e,f = ['a', 'b', 'c', 'd', 'e', 'f']
titles_dict = {'a': 'dicom2nifti',
               'b' : 'dcm2niix',
               'c' : 'dcmstack',
               'd' : 'sitk',
               'e' : 'lifex',
               'f' : 'slicer'}

sys.stdout.write("\n"+f"-"*100)

# For easy reference
input_dir = args.input_dir
output_dir = args.output_dir

# Check the input directory and see if the directories exists
if not(os.path.isdir(input_dir)):
    raise SystemExit("\nProvided input directory does not exist")

# Obtain a list of all of our cases (paths)
cases=[]
for name in os.listdir(input_dir):
    path = os.path.join(input_dir, name)
    if (os.path.isdir(path)):
        cases.append(path)
# If no cases, exit
if not(cases):
    raise SystemExit("\nInput directory is empty")

# For all the cases, check that they have the PET and GT dirs
for case in cases:
    names= os.listdir(case)
    for type in [pet_dir_name, gt_dir_name]:
        if not(type in names):
            raise SystemExit(f"\n{case} does not have {type} directory")
        type_path = os.path.join(case, type)
        if not(os.listdir(type_path)):
            raise SystemExit(f"\n{case} has empty {type} directory")
        
sys.stdout.write(f"\nInput Directory is Suitable")
        

if not(os.path.isdir(output_dir)):
    os.mkdir(output_dir)
    sys.stdout.write(f"\n Output directory at {output_dir}")

sys.stdout.write("\n"+f"-"*100+ "\n")
    
do_a, do_b, do_c, do_d, do_e, do_f=args.a, args.b, args.c, args.d, args.e, args.f 

for name in os.listdir(input_dir):
    path = os.path.join(input_dir, name)
    # The directories will correspond to individual patients
    try:
        if (os.path.isdir(path)):
            pet_dir = os.path.join(path, pet_dir_name)
            output_path = os.path.join(output_dir, name)
            os.mkdir(output_path)
            output_path = os.path.join(output_path, ai4elife_dir_name)
            os.mkdir(output_path)
            sys.stdout.write("\n"+ f"Working on {name}"+ "\n")
            if do_a:
                a_exe = f'dicom2nifti {pet_dir} {output_path}'
                os.system(a_exe)
                sys.stdout.write("\n"+ f"{titles_dict[a]} complete"+ "\n")
                
                # Renaming the file
                for fi in os.listdir(output_path):
                    if fi.endswith(fi_ext):
                        os.rename(os.path.join(output_path, fi),
                                os.path.join(output_path, titles_dict[a]+fi_ext))
            if do_b:
                b_exe = f'dcm2niix -z y -f {titles_dict[b]} -o {output_path} {pet_dir} 1> /dev/null'
                os.system(b_exe)
                sys.stdout.write("\n"+ f"{titles_dict[b]} complete"+ "\n")

            if do_c:
                c_exe = f'dcmstack -d --output-ext {fi_ext} --dest-dir {output_path} -o {titles_dict[c]} {pet_dir}'
                os.system(c_exe)
                sys.stdout.write("\n"+ f"{titles_dict[c]} complete"+ "\n")
            
            if do_d:
                d_exe = f'python {sitk_path} -i {pet_dir} -o {output_path} -f {titles_dict[d]}'
                os.system(d_exe)
                sys.stdout.write("\n"+ f"{titles_dict[d]} complete"+ "\n")
            sys.stdout.write("\n"+f"-"*100+ "\n")
    except: SystemError(f"{name} failed")