 
from features_extractor import FeaturesExtractor
import numpy as np

class NiftExtractorConcat(FeaturesExtractor):
    """ 
    Entrada: 
        Volume Nift: (W, H, B) -> Width, Height and Batch (slices) len
    """ 
    def extract(self, vol, batch_size = 8):
        
        assert vol.ndim == 3, "Volume informado tem shape inválido"

        all_feats = None
        for i in range(0, vol.shape[2], batch_size):
            image = np.stack((vol[:,:,i:i+batch_size],)*3, axis=0)
            image = image.transpose(3, 0, 1, 2)
            current_feats = self._extractor.features(image)
            current_feats = current_feats.data.cpu().numpy().squeeze()
            if all_feats is None:
                all_feats = current_feats.flatten() 
            else:
                all_feats = np.hstack((all_feats, current_feats.flatten()))
 
        return all_feats


    
    