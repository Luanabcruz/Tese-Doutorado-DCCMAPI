from nift_extractor_concat import NiftExtractorConcat
from nifti_loader import NiftLoader
import models.resnet as rnet
import models.vgg as vgg
import models.dpn as dpn
import general_utils as gut 
from preprocessings import NiftResizer

if __name__ == '__main__':
    path = r'E:\DOUTORADO_LUANA\etapa1\bases2D\KiTS19_master'
    k = 6
    out_filename = 'feats_resnet34_concat_interpoleted_k{}'.format(k)
    # 27 é a média de fatias
    slices_number = 27
    resizer = NiftResizer((slices_number, 512,512))
    dataset = NiftLoader(path, resizer)
    # model = vgg.Vgg16()
    model  = rnet.Resnet34()
    # model = dpn.DPN131()
    # print(model.num_features())
    # exit()
    extractor = NiftExtractorConcat(model)

    num_feats = model.num_features() * slices_number

    colums = ['case_name']
    colums.extend(["feat_{}".format(i) for i in range(num_feats)])

    cases_name = gut.cases_name_train_valid(k)
    
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
        
  
    
 
    

    
