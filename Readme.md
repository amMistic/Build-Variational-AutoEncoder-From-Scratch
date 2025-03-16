# Variational Autoencoder (VAE) from Scratch using PyTorch

![image](https://github.com/user-attachments/assets/c71db366-3de9-4a13-afea-29ae9a5ec5a9)

This project implements a **Variational Autoencoder (VAE)** from scratch using PyTorch. The VAE is trained on the MNIST dataset to generate reconstructed digit images with enhanced latent space learning using the reparameterization trick.

## 🚀 Installation
Ensure you have the required dependencies installed:

```bash
pip install torch torchvision matplotlib tqdm
```


## 📋 Project Structure
- **`VariationalAutoEncoder` Class**: Defines the VAE architecture, including encoder, decoder, and the reparameterization trick.
- **Training Loop**: Implements the training process with Binary Cross Entropy (BCE) and KL Divergence loss.
- **Testing Loop**: Evaluates model performance on unseen data.
- **Visualization**: Displays original and reconstructed images.


## 🧠 Model Architecture
### Encoder
- **Input Layer:** 784 → **Hidden Layer:** 400 → **Latent Space Dimension:** 20
- Separate linear layers predict:
  - **Mean ( mean )**
  - **Log Variance (log variance)**

### Decoder
- **Latent Space:** 20 → **Hidden Layer:** 400 → **Output Layer:** 784 (with `nn.Sigmoid()` for pixel values in range [0,1])


## 🔍 Reparameterization Trick
To enable backpropagation through stochastic sampling, the reparameterization trick is used:

[ z = mean + log_variance * epsilon]

Where:
- **epsilon** is a random noise sampled from a normal distribution.
- Element Wise Product operations were performed in between log_variance and epsilon


## 📊 Dataset Preparation
The MNIST dataset is used with transformations applied:
```python
transform = transforms.Compose([
    transforms.ToTensor(),
])
```

Dataloaders are configured for training and testing batches.


## 🔥 Loss Function
**Loss = Binary Cross Entropy (BCE) + KL Divergence**
- **BCE Loss**: Measures reconstruction quality.
- **KL Divergence**: Ensures the learned distribution aligns with the desired Gaussian distribution.

```python
def loss_function(recons_x, x, mean, log_variance):
    loss = F.binary_cross_entropy(recons_x, x.view(-1, 784), reduction='sum')
    kl_divergence = -(0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()))
    return loss + kl_divergence
```


## 🚂 Training
The training loop iterates through multiple epochs:
```python
for epoch in range(1, epochs + 1):
    train(model, optimizer, train_loader, device)
    test(model, test_loader, device)
```


## 📈 Visualization
The results are visualized to compare **Original** vs **Reconstructed** images:

```python
visualize_results(model, test_loader, device)
```


## 📋 Results
- The model successfully reconstructs MNIST digits with clear details.
- The KL divergence term ensures meaningful latent space representations.


## 🙌 Acknowledgements
- **PyTorch** — For providing powerful deep learning libraries.
- **MNIST Dataset** — For digit recognition tasks.

