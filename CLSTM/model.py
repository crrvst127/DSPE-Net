from torch import nn


class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output


def build_model():
    from encoder import Encoder
    from decoder import Decoder
    from model_config import make_parameters
    convlstm_encoder_params, convlstm_decoder_params = make_parameters()
    return ED(Encoder(*convlstm_encoder_params), Decoder(*convlstm_decoder_params))
