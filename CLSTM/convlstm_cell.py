# Original implementation by jhhuang96 (2020).
import torch
import torch.nn as nn


class CLSTM_cell(nn.Module):
    """Convolutional LSTM with group-normalized gates."""
    def __init__(self, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=1):
        if hidden_state is None:
            shape = (inputs.size(1), self.num_features, *inputs.shape[-2:])
            hx = inputs.new_zeros(shape)
            cx = inputs.new_zeros(shape)
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = hx.new_zeros(hx.size(0), self.input_channels, *hx.shape[-2:])
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)
