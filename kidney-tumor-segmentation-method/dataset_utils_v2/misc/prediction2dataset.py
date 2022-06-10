import os
from shutil import copyfile, rmtree

# Dom
# input_dir = r'C:\Users\usuario\Downloads\predictions97'
# output_dir = r'D:\DOUTORADO_LUANA\KiTS19_lua\teste'

# Lua

case_template = "case_{:05d}"
# input_dir = r'D:\Documentos\Projetos\projeto-doutorado\base\predictions97'
# output_dir = r'D:\Documentos\Projetos\projeto-doutorado\base\teste'

input_dir = r'C:\Users\usuario\Desktop\teste challenge\predictions'
output_dir = r'D:\DOUTORADO_LUANA\teste challenge\teste challenge'


def get_num_from_string(string):
    return int(''.join(x for x in string if x.isdigit()))


print('\nShow time...')
for pred_file in os.listdir(input_dir):
    case_code = get_num_from_string(pred_file)
    case = case_template.format(case_code)
    print(pred_file, ' ===> ', case)
    copyfile(os.path.join(input_dir, pred_file),
             os.path.join(output_dir, case, 'prediction.nii.gz'))
