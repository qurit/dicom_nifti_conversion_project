#!/usr/bin/env python3

from helper import *

# Arguments to be passed
argParser = argparse.ArgumentParser(
    prog='PROG',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        Comparison of the results. This must be run after
        create_nifti_files.py and apply_ai4elife.py. 
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

main_dir = args.main_dir

ai_dir = os.path.join(main_dir, ai_dir_name)
temp_dir = os.path.join(main_dir, temp_dir_name)
results_dir = os.path.join(main_dir, results_dir_name)

if not(os.path.isdir(results_dir)):
    os.mkdir(results_dir)

# Dictionary: obtain all the cases
cases = read_dict(ai_dir)
no_cases = len(cases)

# List of all directories in temp_dir
dirs = get_dirs(ai_dir)

for case in track(cases, description="Getting results..."):
    case_dir = os.path.join(results_dir, case)
    os.mkdir(case_dir)
    case_data = get_case_data(dirs, case, ai_dir)
    vol_data = get_vol_data(case, temp_dir)
    get_results(case_dir, case_data, vol_data, case)

make_result_file(ai_dir, results_dir, cases)
get_mae_table_plot(results_dir)