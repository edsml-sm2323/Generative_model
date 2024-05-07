import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['predict_and_display_images']

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def predict_and_display_images(folder_path,
                               file_prefix,
                               start_index,
                               end_index,
                               model_path='best_model.pth'):
    """
        Predicts the next image in a sequence and displays the output images.

        This function takes a range of images,
        processes them through a trained model,
        and displays the predicted images.

        Parameters:
        - folder_path (str): Path to the folder containing the images.
        - file_prefix (str): Prefix of the image files to be processed.
        - start_index (int): Starting index of the image files.
        - end_index (int): Ending index of the image files.
        - model_path (str, optional):
        Path to the trained model file. Default is 'best_model.pth'.

        Returns:
        None:
        This function saves the generated images
        and displays them in a matplotlib plot.
        """
    # Create the model
    input_dim = 64
    hidden_dim = [128, 64]
    kernel_size = (3, 3)
    num_layers = 2
    output_size = (360, 360)

    # Check for GPU
    device = 'cuda' if torch.cuda.device_count() > 0 and torch.cuda.is_available() else 'cpu' # noqa

    # Load the model
    model = ImageSequencePredictor(input_dim,
                                   hidden_dim,
                                   kernel_size,
                                   num_layers, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load and transform images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((360, 360)),
        transforms.ToTensor()
    ])

    image_list = []
    for i in range(start_index, end_index + 1):
        file_path = f"{folder_path}/{file_prefix}{i}.jpg"
        image = Image.open(file_path)
        tensor_image = transform(image)
        image_list.append(tensor_image)

    tensor_images = torch.stack(image_list, dim=0)
    tensor_images = tensor_images.unsqueeze(0)

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_images)

    tensor_data = outputs.squeeze(0)

    # Save and display images
    for i in range(tensor_data.size(0)):
        single_channel_image = tensor_data[i, :, :].squeeze()
        tensor_to_pil = transforms.ToPILImage()(single_channel_image)
        tensor_to_pil.save(f"generate_{i + 1}.jpg")

    gray_images = [tensor_data[i, :, :].squeeze() for i in range(tensor_data.size(0))] # noqa
    fig, axs = plt.subplots(1, len(gray_images), figsize=(12, 4))
    for i, img in enumerate(gray_images):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'generate {i + 1}')
    plt.show()


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 kernel_size, bias=True):
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
                                             self.hidden_dim, dim=1)
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
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1] # noqa
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
    def __init__(self, input_dim, hidden_dim,
                 kernel_size, num_layers, output_size=(128, 128)):
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
                               stride=2,
                               padding=1),  # sencod layer
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3,
                               stride=2,
                               padding=1),  # third layer
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3,
                               stride=2,
                               padding=1),  # Forth layer
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1*3,
                               kernel_size=2, stride=2),  # final output
            nn.Sigmoid()
            )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.conv_layers(x)

        # Rsize dims
        x = x.view(batch_size,
                   timesteps, -1, H // 4, W // 4)
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

# Example usage
# predict_and_display_images('tst', 'tst_', 242, 251,'best_model.pth')