#!/usr/bin/env python

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
            -lifex_slicer_dir
            -dcmrtstruct2nii
            -lifex
            -rt_utils
            -slicer
        '''))
argParser.add_argument("-m", "--main_dir", help="path to main directory", type=str, required=True)

args = argParser.parse_args()

main_dir = args.main_dir

for gt_key in gt_keys:
    gt_name = gt_dict[gt_key]
    sys.stdout.write(f"Working on {gt_name}")
    gt_dir = os.path.join(main_dir, gt_name)
    temp_dir = os.path.join(gt_dir, temp_dir_name)
    ai4elife_dir = os.path.join(gt_dir, ai_dir_name)
    results_dir = os.path.join(gt_dir, results_dir_name)
    
    if not(os.path.isdir(results_dir)):
        os.mkdir(results_dir)

    # Dictionary: obtain all the cases
    cases = read_dict(ai4elife_dir)
    no_cases = len(cases)

    # List of all directories in temp_dir
    dirs = get_dirs(ai4elife_dir)

    for case in track(cases, description="Making Results Files..."):
        case_dir = os.path.join(results_dir, case)
        os.mkdir(case_dir)
        case_data = get_case_data(dirs, case, ai4elife_dir)
        vol_data = get_vol_data(case, temp_dir)
        get_results(case_dir, case_data, vol_data, case)

    make_result_file(temp_dir, results_dir, cases)
combine_result_files(main_dir)
get_overall_results(main_dir)