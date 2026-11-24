from collections import OrderedDict
from convlstm_cell import CLSTM_cell


# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
def make_parameters():
    convlstm_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
        ],

        [
            CLSTM_cell(input_channels=16, filter_size=5, num_features=64),
            CLSTM_cell(input_channels=64, filter_size=5, num_features=96),
            CLSTM_cell(input_channels=96, filter_size=5, num_features=96)
        ]
    ]

    convlstm_decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
            OrderedDict({
                'conv3_leaky_1': [64, 16, 3, 1, 1],
                'conv4_leaky_1': [16, 1, 1, 1, 0]
            }),
        ],

        [
            CLSTM_cell(input_channels=96, filter_size=5, num_features=96),
            CLSTM_cell(input_channels=96, filter_size=5, num_features=96),
            CLSTM_cell(input_channels=96, filter_size=5, num_features=64),
        ]
    ]
    return convlstm_encoder_params, convlstm_decoder_params
