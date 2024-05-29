import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义Self-attention层
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention)
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class ConvLSTMCell(nn.Module):
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
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
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


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            # add attention
            SelfAttention(32),

            

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64, 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            # add attention
            SelfAttention(32),
           
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ImageSequencePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_size=(256, 256)):
        super(ImageSequencePredictor, self).__init__()
        self.output_size = output_size

        # Convolutional Autoencoder
        self.cae = CAE()

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


    def forward(self, x):
        print(x.shape)
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cae.encoder(x)
        
        x = x.view(batch_size, timesteps, -1, H // 4, W // 4)
        
        layer_output_list, last_state_list = self.conv_lstm(x)

        x = last_state_list[-1][0]  # The final hidden state
        
        x = self.cae.decoder(x)
        
        return x
