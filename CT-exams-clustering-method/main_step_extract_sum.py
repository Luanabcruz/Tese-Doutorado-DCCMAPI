from nift_extractor_sum import NiftExtractorSum
from nift_extractor_concat import NiftExtractorConcat
from nifti_loader import NiftLoader
import models.resnet as rnet
import models.vgg as vgg
import general_utils as gut 
from preprocessings import NiftResizer

if __name__ == '__main__':
    path = r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'
    out_filename = 'feats_vgg11_sum_interpoleted_trash'
    # 27 é a média de fatias
    slices_number = 27
    resizer = NiftResizer((slices_number, 512,512))
    dataset = NiftLoader(path, resizer)
    model = vgg.Vgg11()
    extractor = NiftExtractorSum(model)

    num_feats = model.num_features() # isso aqui só é pra o sum
    
    colums = ['case_name']
    colums.extend(["feat_{}".format(i) for i in range(num_feats)])

    cases_name = gut.cases_name_train_valid()
    
    cases_feats = []
    
    for case_name in cases_name:
        case_data = dataset.getCaseDataByName(case_name)
        feats = extractor.extract(case_data, batch_size = 32)

        case_feat = {
            "case_name": case_name,
        }

        for i in range(len(feats)):
            case_feat["feat_{}".format(i)] = feats[i]

        cases_feats.append(case_feat)
        gut.save_to_csv(colums, cases_feats, out_filename)
        
  
    
 
    

    
