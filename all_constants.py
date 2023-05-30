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
case_columns = ["Case", "Conversion Type", "Sagittal Dice Score", "Coronal Dice Score", "Mean Absolute Error"]
all_columns = case_columns+["Predicted Sagittal TMTV", "Ground Truth Sagittal TMTV",
                 "Predicted Coronal TMTV", "Ground Truth Sagittal TMTV",
                 "Predicted Total TMTV", "Ground Truth Total TMTV"]
csv_ext = '.csv'
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
cut_dict = {0:'Sagittal', 1: 'Coronal'}

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