import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ConvLSTMCell', 'ConvLSTM', 'ImageSequencePredictor']

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class ConvLSTMCell(nn.Module):
    """
        A convolutional LSTM cell module.

        Parameters:
        - input_dim (int): The number of input channels.
        - hidden_dim (int): The number of hidden channels.
        - kernel_size (tuple): The size of the convolutional kernel.
        - bias (bool, optional):
        If True, adds a learnable bias to the output. Default is True.

        The forward pass of the cell takes in an input tensor
        and the previous state and outputs the next hidden state and
        cell state.
        """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,
                                             self.hidden_dim,
                                             dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size,
                            self.hidden_dim,
                            height, width,
                            device=device),
                torch.zeros(batch_size,
                            self.hidden_dim,
                            height, width,
                            device=device))


class ConvLSTM(nn.Module):
    """
       A Convolutional LSTM module, consisting of multiple ConvLSTMCell layers.

       Parameters:
       - input_dim (int): The number of input channels.
       - hidden_dim (list of int):
       A list defining the number of hidden channels per layer.
       - kernel_size (tuple): The size of the convolutional kernel.
       - num_layers (int): The number of layers in the ConvLSTM.
       - batch_first (bool, optional):
       f True, the input and output tensors
        are provided as (batch, seq, channel, height, width). Default is True.
       - bias (bool, optional):
       If True, adds a learnable bias to the output. Default is True.
       - return_all_layers (bool, optional):
       If True, returns the hidden state for all layers. Default is False.

       The forward pass takes in an input tensor
       and an optional hidden state
       and returns the output tensor and
       the last state of each layer.
       """
    def __init__(self, input_dim, hidden_dim, kernel_size,
                 num_layers, batch_first=True,
                 bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim \
                if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # Convert to (batch, seq_len, channels, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            # Initialize hidden state if not provided
            hidden_state = [self.cell_list[i].init_hidden(b, (h, w))
                            for i in range(self.num_layers)]

        current_input = input_tensor
        layer_output_list = []
        last_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(current_input.size(1)):
                h, c = self.cell_list[layer_idx](current_input[:, t, :, :, :],
                                                 (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            current_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list


class ImageSequencePredictor(nn.Module):
    """
        An image sequence prediction model using Convolutional LSTM.

        Parameters:
        - input_dim (int):
        The number of input channels.
        - hidden_dim (list of int):
        A list defining the number
        of hidden channels per layer in the ConvLSTM.
        - kernel_size (tuple):
        The size of the convolutional kernel for ConvLSTM.
        - num_layers (int):
        The number of layers in the ConvLSTM.
        - output_size (tuple, optional):
        The size of the output images. Default is (128, 128).

        The model consists of a simple convolutional layer,
        followed by a ConvLSTM layer
        and transposed convolutional layers for upsampling.
        The forward pass takes in a batch of image sequences and
        outputs the predicted image sequences.
        """
    def __init__(self, input_dim, hidden_dim, kernel_size,
                 num_layers, output_size=(128, 128)):
        super(ImageSequencePredictor, self).__init__()
        self.output_size = output_size

        # Simple CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ConvLSTM layer
        self.conv_lstm = ConvLSTM(
            input_dim=64,  # Simple CNN output
            hidden_dim=hidden_dim,  # Hidden dims
            kernel_size=kernel_size,  # kernel size
            num_layers=num_layers,  # layers nmber
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # transpose conv to generate images
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3,
                               stride=2, padding=1),  # sencod layer
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3,
                               stride=2, padding=1),  # third layer
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3,
                               stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1*3, kernel_size=2,
                               stride=2),
            nn.Sigmoid()
            )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.conv_layers(x)

        # Rsize dims
        x = x.view(batch_size, timesteps, -1, H // 4, W // 4)
        layer_output_list, last_state_list = self.conv_lstm(x)

        # Use the final CNN time
        x = last_state_list[-1][0]  # The final states

        # Change the size
        x = self.trans_conv(x)

        # resize the output
        x = F.interpolate(x, size=self.output_size,
                          mode='bilinear', align_corners=False)
        x = x.unsqueeze(2)

        return x