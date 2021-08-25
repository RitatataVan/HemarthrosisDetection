"""
    Definition of Models

    * Implemented Models:
        - ResNet 18
        - EfficientNet B4
        - EfficientNet B4 - Transfer Learning from Knees
"""

import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from hyperparameters import parameters as params


def resnet18():
    """
        efficientnet ResNet 18 model definition.
    """
    out_features = params['out_features']

    model = models.resnet18(pretrained=True)

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # New output layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features)

    return model


def efficientnet():
    """
        efficientnet EfficientNet B4 model definition.
    """
    out_features = params['out_features']

    model = EfficientNet.from_pretrained('efficientnet-b4')

    # To freeze layers
    # for param in model.parameters():
    #    param.requires_grad = False

    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=out_features)

    return model


def efficientnet_knees():
    ''' EfficientNet B4 trained on knees. '''

    out_features = params['out_features']

    model = efficientnet()

    weights = 'adteff-fold5-epoch=05.ckpt'

    checkpoint = torch.load(weights, map_location='cuda')
    #model.load_state_dict(checkpoint['state_dict'])

    for key in checkpoint['state_dict']:
        new_key = key.replace("model.", "")
        checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)



    print(checkpoint['state_dict'].keys())
    exit()

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=out_features)

    return model

