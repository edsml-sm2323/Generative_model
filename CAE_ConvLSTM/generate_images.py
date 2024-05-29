import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = ["generate"]


def generate(model, test_data, background_data, obs_data, start, num_inputs, device):
    """
    Generate output images using the trained model and compare them with the target images.

    Parameters:
    - model (nn.Module): The trained model.
    - test_data (str): Path to the .npy file containing test images.
    - background_data (str): Path to the .npy file containing background images.
    - obs_data (str): Path to the .npy file containing observation images.
    - start (int): The starting index for input images in the test dataset.
    - num_inputs (int): The number of input images required by the model.
    - device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
    - Tuple of output images for test, background, and observation datasets.
    """
    model.eval()  # Set model to evaluation mode

    # Load data
    test_data = np.load(test_data)
    background_data = np.load(background_data)
    obs_data = np.load(obs_data)

    def process_test_data(data, start, num_inputs):
        input_images = data[start : start + num_inputs * 9 : 9]
        target_image = data[start + num_inputs * 9]
        return input_images, target_image

    def process_obs_background_data(data, num_inputs):
        input_images = data[:num_inputs]
        target_image = data[num_inputs]
        return input_images, target_image

    def plot_images(input_images, output_image, target_image, title_prefix, row):
        for i in range(num_inputs):
            plt.subplot(3, num_inputs + 2, row * (num_inputs + 2) + i + 1)
            plt.imshow(input_images[0, i, 0].cpu().numpy(), cmap="gray")
            plt.title(f"{title_prefix} Input {i + 1}")
            plt.axis("off")

        plt.subplot(3, num_inputs + 2, row * (num_inputs + 2) + num_inputs + 1)
        plt.imshow(output_image, cmap="gray")
        plt.title(f"{title_prefix} Generated")
        plt.axis("off")

        plt.subplot(3, num_inputs + 2, row * (num_inputs + 2) + num_inputs + 2)
        plt.imshow(target_image, cmap="gray")
        plt.title(f"{title_prefix} Target")
        plt.axis("off")

    def generate_and_plot_test(data, start, num_inputs, title_prefix, row):
        input_images, target_image = process_test_data(data, start, num_inputs)
        input_images = (
            torch.tensor(input_images, dtype=torch.float32).unsqueeze(1).to(device)
        )
        target_image = (
            torch.tensor(target_image, dtype=torch.float32).unsqueeze(0).to(device)
        )
        input_images = input_images.unsqueeze(0)  # Shape: (1, num_inputs, 1, H, W)

        with torch.no_grad():
            output = model(input_images)

        output_image = output.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (H, W)
        target_image = target_image.squeeze(0).cpu().numpy()  # Shape: (H, W)

        mse = np.mean((output_image - target_image) ** 2)
        print(f"{title_prefix} Mean Squared Error (MSE): {mse:.4f}")

        plot_images(input_images, output_image, target_image, title_prefix, row)
        return output_image

    def generate_and_plot_obs_background(data, num_inputs, title_prefix, row):
        input_images, target_image = process_obs_background_data(data, num_inputs)
        input_images = (
            torch.tensor(input_images, dtype=torch.float32).unsqueeze(1).to(device)
        )
        target_image = (
            torch.tensor(target_image, dtype=torch.float32).unsqueeze(0).to(device)
        )
        input_images = input_images.unsqueeze(0)  # Shape: (1, num_inputs, 1, H, W)

        with torch.no_grad():
            output = model(input_images)

        output_image = output.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (H, W)
        target_image = target_image.squeeze(0).cpu().numpy()  # Shape: (H, W)

        mse = np.mean((output_image - target_image) ** 2)
        print(f"{title_prefix} Mean Squared Error (MSE): {mse:.4f}")

        plot_images(input_images, output_image, target_image, title_prefix, row)
        return output_image

    plt.figure(figsize=(15, 15))

    test_output = generate_and_plot_test(test_data, start, num_inputs, "Test", 0)
    background_output = generate_and_plot_obs_background(
        background_data, num_inputs, "Background", 1
    )
    obs_output = generate_and_plot_obs_background(obs_data, num_inputs, "Obs", 2)

    plt.show()

    return test_output, background_output, obs_output
