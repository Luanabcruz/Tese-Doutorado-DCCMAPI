deepv3_dpn131_sem_augm= {
        'name': 'Deeplabv3_sem_augm',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/deeplabv3plus_25D_natalv7_dpn131_sem_aug.pt',
        'postprocessing': False
}

deepv3_dpn131_dao_sem_pre = {
        'name': 'Deeplabv3_dpn131_DAO',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/deeplabv3plus_25D_natalv7_dpn131_data_offline.pt',
        'postprocessing': False
}
deepv3_dpn131_sem_pre = {
        'name': 'Deeplabv3_dpn131',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/TW_V3_0_deep_plus_dpn131.pt',
        'postprocessing': False
}




# Backbones for UNETS 
unet_default_com_pre = {
        'name': 'Unet_Default',
        'weights': r'../pesos_luanet/tumor/unet_25D_natalv7_classica.pt',
        'postprocessing': True
}

unet_resunet101_com_pre = {
        'name': 'Unet_Resnet101',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/resunet101_25D_natal_v7_v2_nerfado.pt',
        'postprocessing': True
}

# Backbones for Deeplabv3 

deepv3_xception_com_pre = {
        'name': 'Deeplabv3_xception',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/deeplabv3plus_25D_natalv7_xception.pt',
        'postprocessing': True
}

deepv3_resnet101_com_pre = {
        'name': 'Deeplabv3_resnet101',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/deeplabv3plus_25D_natalv7_resnet101_nerfado_v3.pt',
        'postprocessing': True
}

deepv3_dpn131_com_pre = {
        'name': 'Deeplabv3_dpn131',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/TW_V3_0_deep_plus_dpn131.pt',
        'postprocessing': True
}

deepv3_dpn131_sem_balanceamento_com_pre = {
        'name': 'Deeplabv3_dpn131_sem_balanc_com_pre',
        'weights': r'../pesos_luanet/tumor/pesos_paper_revision/deeplabv3plus_25D_natalv7_dpn131_sem_balanceamento_fatias.pt',
        'postprocessing': True
}


def get_params():

    return [
        # comentadas pq já foram executadas
        deepv3_dpn131_sem_augm,
        deepv3_dpn131_dao_sem_pre,
        deepv3_dpn131_sem_pre,
        deepv3_dpn131_com_pre,
        unet_default_com_pre, 
        unet_resunet101_com_pre,
        deepv3_xception_com_pre,
        deepv3_resnet101_com_pre, #// 
        deepv3_dpn131_sem_balanceamento_com_pre,
    ]



