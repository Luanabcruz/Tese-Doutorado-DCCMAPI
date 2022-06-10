import torch
import numpy as np
from torchvision import transforms

# batches of C channel image (B, C, W, H)
def transform_input(image: np) -> torch.Tensor:
    # imagenet normalization image to deep features
    transform = transforms.Compose([           
        transforms.ToTensor(),                     
        transforms.Normalize(                      
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]                  
        )
    ])

    new_image = torch.zeros(image.shape)
    image = image.transpose(0, 2, 3, 1)
    
    for batch_idx in range(0, new_image.shape[0]):
        new_image[batch_idx] = transform(image[batch_idx])
    return new_image

def getDevice() -> str:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
