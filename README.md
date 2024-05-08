# Generative model
According to different data and different task goals, this repositories lists four generative models I built.  
Including VAE, GAN, CNN-LSTM and Unet-SelfAttention.

## VAE model
The VAE model consists of two main components: an `encoder` and a `decoder`. 

The encoder takes input data, such as images, and maps it to a lower-dimensional latent space representation. The decoder then takes samples from this latent space and reconstructs the original data.  

During training, the VAE aims to minimize a loss function that measures the difference between the input data and the reconstructed data, while also encouraging the distributions outputted by the encoder and decoder to be similar to a predefined prior distribution (often a simple Gaussian distribution).
<div align="center">
  <img src="VAE/vae_structure.png" width="500" />
</div> 

### Task

### Dataset

### Model structure

### Important concept


## GAN model

## CNN-LSTM model

## Unet-Self Attention model

