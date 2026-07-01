"""
TimeGAN (Temporal Generative Adversarial Network) for Synthetic Market Data Generation.
Generates realistic financial time series for stress testing and augmenting training data.
Capable of creating black swan scenarios and rare market events not present in historical data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
import numpy as np

class Generator(nn.Module):
    """Generates synthetic time series from random noise."""
    def __init__(self, input_size: int, hidden_size: int, seq_len: int, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Random noise (batch, seq_len, input_size)
        Returns:
            Synthetic sequence (batch, seq_len, input_size)
        """
        out, _ = self.lstm(z)
        return self.fc(out)

class Discriminator(nn.Module):
    """Distinguishes real from synthetic time series."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Take last time step for classification
        return self.fc(out[:, -1, :])

class Embedder(nn.Module):
    """Maps real data to latent space for better GAN training stability."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Recoverer(nn.Module):
    """Maps latent space back to data space."""
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

class TimeGAN:
    """
    Complete TimeGAN framework for synthetic time series generation.
    Includes embedding/recovery for improved stability and temporal consistency.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 seq_len: int,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        # Initialize networks
        self.embedder = Embedder(input_size, hidden_size).to(self.device)
        self.recoverer = Recoverer(hidden_size, input_size).to(self.device)
        self.generator = Generator(hidden_size, hidden_size, seq_len).to(self.device)
        self.discriminator = Discriminator(hidden_size, hidden_size).to(self.device)
        
        # Optimizers
        self.opt_embedder = optim.Adam(self.embedder.parameters(), lr=learning_rate)
        self.opt_recoverer = optim.Adam(self.recoverer.parameters(), lr=learning_rate)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def train_embedding(self, real_data: torch.Tensor, epochs: int = 1000):
        """Train embedder and recoverer to minimize reconstruction error."""
        self.embedder.train()
        self.recoverer.train()
        
        for epoch in range(epochs):
            self.opt_embedder.zero_grad()
            self.opt_recoverer.zero_grad()
            
            # Embed and recover
            h = self.embedder(real_data)
            recovered = self.recoverer(h)
            
            # Reconstruction loss
            loss = self.mse_loss(recovered, real_data)
            loss.backward()
            
            self.opt_embedder.step()
            self.opt_recoverer.step()
            
            if epoch % 200 == 0:
                print(f"Embedding Epoch {epoch}, Loss: {loss.item():.4f}")
                
    def train_gan(self, real_data: torch.Tensor, epochs: int = 1000):
        """Train generator and discriminator in adversarial fashion."""
        for epoch in range(epochs):
            # Train discriminator
            self.discriminator.train()
            self.generator.train()
            
            # Real data
            h_real = self.embedder(real_data).detach()
            d_real = self.discriminator(h_real)
            
            # Fake data
            z = torch.randn_like(h_real)
            fake = self.generator(z)
            d_fake = self.discriminator(fake.detach())
            
            # Discriminator loss
            d_loss_real = self.bce_loss(d_real, torch.ones_like(d_real))
            d_loss_fake = self.bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = d_loss_real + d_loss_fake
            
            self.opt_discriminator.zero_grad()
            d_loss.backward()
            self.opt_discriminator.step()
            
            # Train generator
            z = torch.randn_like(h_real)
            fake = self.generator(z)
            d_fake_gen = self.discriminator(fake)
            
            g_loss = self.bce_loss(d_fake_gen, torch.ones_like(d_fake_gen))
            
            self.opt_generator.zero_grad()
            g_loss.backward()
            self.opt_generator.step()
            
            if epoch % 200 == 0:
                print(f"GAN Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
                
    def generate_synthetic(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic time series."""
        self.generator.eval()
        z = torch.randn(num_samples, self.seq_len, self.hidden_size, device=self.device)
        
        with torch.no_grad():
            fake_latent = self.generator(z)
            synthetic = self.recoverer(fake_latent)
            
        return synthetic.cpu()
    
    def generate_stress_scenario(self, base_data: torch.Tensor, severity: float = 2.0) -> torch.Tensor:
        """
        Generate extreme market scenarios (black swans).
        Adds controlled perturbations to latent space.
        """
        self.generator.eval()
        
        # Embed base data
        with torch.no_grad():
            h_base = self.embedder(base_data.to(self.device))
            
        # Add extreme noise in latent space
        noise = torch.randn_like(h_base) * severity
        h_stress = h_base + noise
        
        # Generate stressed scenario
        with torch.no_grad():
            fake_latent = self.generator(h_stress)
            stressed = self.recoverer(fake_latent)
            
        return stressed.cpu()

# Verification
if __name__ == "__main__":
    print("Initializing TimeGAN...")
    
    input_size = 5  # e.g., OHLCV
    hidden_size = 32
    seq_len = 20
    
    gan = TimeGAN(input_size, hidden_size, seq_len)
    
    # Create dummy real data
    real_data = torch.randn(64, seq_len, input_size)
    
    # Quick training test
    print("\n--- Training Embedding ---")
    gan.train_embedding(real_data, epochs=100)
    
    print("\n--- Training GAN ---")
    gan.train_gan(real_data, epochs=100)
    
    # Generate synthetic data
    print("\n--- Generating Synthetic Data ---")
    synthetic = gan.generate_synthetic(num_samples=10)
    print(f"Synthetic data shape: {synthetic.shape}")
    
    # Generate stress scenario
    print("\n--- Generating Stress Scenario ---")
    stress = gan.generate_stress_scenario(real_data[:5], severity=3.0)
    print(f"Stress scenario shape: {stress.shape}")
    print("TimeGAN Initialization Successful.")
