#!/usr/bin/env python3

from helper import *

# Arguments to be passed
argParser = argparse.ArgumentParser(
    prog='PROG',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        Comparison of the results. This must be run after
        create_nifti_files.py and apply_ai4elife.py. 
        '''))
argParser.add_argument("-i", "--input_dir", help="path to dir with ai4elife output ", type=str, required=True)
argParser.add_argument("-o", "--output_dir", help="path to dir where results will be saved", type=str, required=True)
argParser.add_argument("-t", "--temp_dir", help="path to dir with inputs for ai4elife", type=str, required=True)

args = argParser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
temp_dir = args.temp_dir

if not(os.path.isdir(output_dir)):
    os.mkdir(output_dir)

# Dictionary: obtain all the cases
cases = read_dict(input_dir)
no_cases = len(cases)

# List of all directories in temp_dir
dirs = get_dirs(input_dir)

for case in track(cases, description="Getting results..."):
    case_dir = os.path.join(output_dir, case)
    os.mkdir(case_dir)
    case_data = get_case_data(dirs, case, input_dir)
    vol_data = get_vol_data(case, temp_dir)
    get_results(case_dir, case_data, vol_data, case)

make_result_file(input_dir, output_dir, cases)
get_mae_table_plot(output_dir)