import os
import torch

from trainer import train_model

from utils import create_nested_dir, get_criterion_by_name, get_segmentation_model, get_data_loaders, get_log_path
from kidney_dataset_utils.misc.split_kits19 import load_dpa_dist
from kidney_utils.metrics import mean_dice_coef, mean_dice_coef_remove_empty

import lua_enums as l_enum

from get_env import datasets_path

def run_nn(dpa_dist):
    """
        Version requirements:
            PyTorch Version:  >1.2.0
            Torchvision Version:  >0.4.0a0+6b959ee
    """

    """
        Parameters to execute the method
    """
    
    root_dir = datasets_path()['reconst25D']
    
    epochs = 100
    batch_size = 7
    # Filename of the final model weigths
    weight_filename = "weights_final.pt"
    loss_name = "dice"
    # the distribution of the data set created in the spreadsheet
    dataset_spreed_id = "dpa_runs_reconstruction"
    # The folder that will contain weights and log file.\
    cnn = 'unet'
    backbone = 'resnet101'

    data_aug = l_enum.DataAug.ONLINE

    in_channels = 3

    log_path = get_log_path(data_aug, cnn, loss_name,
                            backbone, dataset_spreed_id, in_channels,sufix = dpa_dist)
    print("###\n")
    print("| {} => {} | Data Aug: {} ".format(
        cnn.upper(), backbone.upper(), data_aug.value))
    print("| Loss: {}\n".format(loss_name))
    print("| Canais: {}\n".format(in_channels))
    print("###")

    """
        Main 
    """

    create_nested_dir(log_path)

    # Loads the distribution of the cases according to the spreadsheet
    cases = load_dpa_dist(dpa_dist)

    # Create the dataloader

    dataloaders = get_data_loaders(
        data_aug, cases, root_dir, batch_size)

    model = get_segmentation_model(backbone, cnn, in_channels=in_channels)
    model.train()

    # Load the loss object by name
    criterion = get_criterion_by_name(loss_name)

    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.9
    )

    # Specify the evalutation metrics
    metrics = {'dice': mean_dice_coef,
               'dice_lesao': mean_dice_coef_remove_empty}

    train_model(model, criterion, dataloaders,
                optimizer, exp_lr_scheduler, bpath=log_path, metrics=metrics, num_epochs=epochs)

    # Save the trained model
    torch.save(model, os.path.join(log_path, weight_filename))
    print('\n\n ### ===> Training finished sucessfully!\n\n')


if __name__ == '__main__':

    for file in os.listdir('./configuration_files/dpa'):
        print("Running: ", file)
        run_nn(file)
