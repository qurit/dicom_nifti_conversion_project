# Global constants
pet_dir_name = 'PET'
gt_dir_name = 'GT'
fi_ext = '.nii.gz'
sitk_path = './sitk.py'
ai4elife_pet_dir_name = 'pet'
ai4elife_gt_dir_name = 'gt'
pred_dir_name = 'predicted_data'
dict_name = 'cases'
dict_ext = '.txt'
pet_end = 'pet.nii'
pred_end = 'predicted.nii'
gt_end = 'ground_truth.nii'
mask_end = 'predicted.nii'
fontsize = 24
no_rt_struct_prefix = 'PETCT'
no_rt_struct_conv_name = "dcm2nii_mask"
case_columns = ["Case", "Conversion Type", "Sagittal Dice Score", "Coronal Dice Score", "Mean Absolute Error"]
all_columns = case_columns+["Predicted Sagittal TMTV", "Ground Truth Sagittal TMTV",
                 "Predicted Coronal TMTV", "Ground Truth Sagittal TMTV",
                 "Predicted Total TMTV", "Ground Truth Total TMTV"]
combined_columns = ["GT Conversion Method"] + all_columns
csv_ext = '.csv'
results_fi_name = 'results'+csv_ext
gt_csv_fi = 'surrogate_ground_truth.csv'
pred_csv_fi = 'surrogate_predicted.csv'

keys = ['a', 'b', 'c', 'd', 'e', 'f']
a,b,c,d,e,f = keys
titles_dict = {
    a : 'dicom2nifti',
    b : 'dcm2niix',
    c : 'dcmstack',
    d : 'sitk',
    e : 'lifex',
    f : 'slicer'}

cuts = [0,1]
cut_dict = {
    0:'Sagittal',
    1: 'Coronal',
    2: 'Total'}

GT='gt'
PRED='pred'
PID='pid'
SAG='sag'
COR='cor'
TOT='tot'
fi_keys = [GT, PRED]

mir_required = {
    titles_dict[a] : True,
    titles_dict[b] : True, 
    titles_dict[c] : True, 
    titles_dict[d] : False,
    titles_dict[e] : False,
    titles_dict[f] : False
}

z,u,v,x = 'z', 'u', 'v','x'
gt_keys = z,u,v,x

gt_dict = {
    x : 'rt_utils', 
    # y : 'pyradise',
    z : 'dcmrtstruct2nii',
    u : 'lifex',
    v : 'slicer'
}

bad_words = ['SUV', 'Liver', 'liver']

lifex_gt_path = "/home/jhubadmin/qurit/dicom2nifti/gt_exp/lifex_ground_truth"
slicer_gt_path = "/home/jhubadmin/qurit/dicom2nifti/gt_exp/slicer_ground_truth"

box_plot_keys = [a,b,c]
box_plot_fi_name = "box_plot.png"

mae_table_columns = ["Conversion Type", "Averaged Mean Absolute Error"]
mae_table_fi_name = "mae_results.csv"

combined_results_fi = "combined_results.csv"

gt_conv_csv_index = combined_columns[0]
sag_dice_csv_index = combined_columns[3]
cor_dice_csv_index = combined_columns[4]
pred_sag_tmtv_csv_index = combined_columns[6]
gt_sag_tmtv_csv_index = combined_columns[7]
pred_cor_tmtv_csv_index = combined_columns[8]
gt_cor_tmtv_csv_index = combined_columns[9]
pred_tot_tmtv_csv_index = combined_columns[10]
gt_tot_tmtv_csv_index = combined_columns[11]


box_plot_x_axis = "Conversion Method"


CONV_TYPE, SAG_DICE, COR_DICE, SAG_TMTV, COR_TMTV, TOT_TMTV = 'p', 'q', 'r', 's', 't', 'u'

gt_columns = ["Mean Sagittal Dice", "Mean Coronal Dice",
              "Mean Absolute Error Sagittal TMTV",
              "Mean Absolute Error Coronal TMTV",
              "Mean Absolute Error Total TMTV"]

gt_csv_fi = "gt_comparison_results.csv"

data_dir_name = "data_dir"
temp_dir_name = "temp_dir"
ai_dir_name = "ai_dir"
results_dir_name = "results_dir"
lifex_slicer_dir_name = "lifex_slicer_dir"



