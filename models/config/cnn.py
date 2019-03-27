#!/usr/bin/Python
# -*- coding: utf-8 -*-

LeNet = {
    'model': [
        {
            'name': 'conv_1',
            'type': 'conv',
            'k_size': [5, 5],
            'bn': True,
            'filter_out': 6,
        },
        {
            'name': 'pool_2',
            'type': 'pool',
            'k_size': 2,
        },
        {
            'name': 'conv_3',
            'type': 'conv',
            'k_size': [5, 5],
            'bn': True,
            'filter_out': 16,
        },
        {
            'name': 'pool_4',
            'type': 'pool',
            'k_size': 2,
        },
        {
            'name': 'conv_5',
            'type': 'conv',
            'k_size': [5, 5],
            'bn': True,
            'filter_out': 120,
        },
        {
            'name': 'fc_6',
            'type': 'fc',
            'bn': True,
            'filter_out': 20,
        },
        {
            'name': 'dropout',
            'type': 'dropout',
        },
        {
            'name': 'fc_7',
            'type': 'fc',
            'bn': True,
            'filter_out': 2,
            'activate': False,
        },
    ],
    'input_shape': [None, 10, 20, 1],
    'use_bn': True,
}

CNN_1D = {
    'model': [
        {
            'name': 'conv_1_1',
            'type': 'conv',
            'shape': [1, 6],
            'k_size': [1, 5],
            'bn': True,
            'stride': [1, 1, 1, 1],
        },
        {
            'name': 'conv_1_2',
            'type': 'conv',
            'shape': [6, 12],
            'k_size': [1, 5],
            'bn': True,
            'stride': [1, 1, 2, 1],
        },
        {
            'name': 'conv_2_1',
            'type': 'conv',
            'shape': [12, 12],
            'k_size': [1, 5],
            'bn': True,
            'stride': [1, 1, 1, 1],
        },
        {
            'name': 'conv_2_2',
            'type': 'conv',
            'shape': [12, 32],
            'k_size': [1, 5],
            'bn': True,
            'stride': [1, 1, 2, 1],
        },
        {
            'name': 'fc_3',
            'type': 'fc',
            'filter_out': 32,
            'trainable': True,
            'BN': True,
        },
        {
            'name': 'dropout',
            'type': 'dropout',
        },
        {
            'name': 'softmax',
            'type': 'fc',
            'filter_out': 2,
            'activate': False,
        },
    ],
    'input_shape': [None, 1, 200, 1],
    'use_bn': True,
}
