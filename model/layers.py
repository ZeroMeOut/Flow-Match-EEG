import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log as ln

class LIU(nn.Module):
    def __init__ (self):
        super(LIU, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1)
        self.norm = nn.InstanceNorm1d(64)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        if x.size(-1) > 1:
            x = self.norm(x)
        return x

## From  https://arxiv.org/pdf/2011.01557#page=0.99
class TADELayer(nn.Module):
    def __init__(self, activation_channels=64, conditioning_channels=64):
        super(TADELayer, self).__init__()
        self.instance_norm = nn.InstanceNorm1d(activation_channels, affine=False)

        self.conv_gamma = nn.Conv1d(conditioning_channels, activation_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv1d(conditioning_channels, activation_channels, kernel_size=3, padding=1)

    
    def forward(self, activations, conditioning):
        batch_size, _, time_steps = activations.shape

        normalized_activations = self.instance_norm(activations)

        upsampled_conditioning = F.interpolate(
            conditioning,  
            size=(time_steps,),  
            mode='linear', 
            align_corners=False
        )  

        gamma = self.conv_gamma(upsampled_conditioning)  
        beta = self.conv_beta(upsampled_conditioning) 

        # γ * normalized_content + β
        modulated_activations = gamma * normalized_activations + beta

        return modulated_activations

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super(EncoderLayer, self).__init__()
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.1, 
            activation='relu',
            batch_first=True  
        )
    
    def forward(self, x):
        # x shape: [batch_size,  embed_dim, seq_length]
        # TransformerEncoderLayer expects input shape: [batch_size, seq_length, embed_dim]
        x = x.permute(0, 2, 1)
        x = self.TransformerEncoderLayer(x)
        # Convert back to [batch_size,  embed_dim, seq_length]
        x = x.permute(0, 2, 1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding.unsqueeze(-1)

## Test Steps
if __name__ == "__main__":

    ## Test LIU Layer
    input_tensor = torch.randn(64, 1, 1250)
    noise_level = torch.randn(64, 1)

    liu_layer = LIU()
    output_tensor = liu_layer(input_tensor)
    print("LIU Layer Output Shape:", output_tensor.shape)

    ## Test Transformer Encoder Layer
    encoder_layer = EncoderLayer()
    encoder_output = encoder_layer(output_tensor)
    print("Encoder Layer Output Shape:", encoder_output.shape)

    ## Test Positional Encoding
    positional_encoding = PositionalEncoding(dim=64)
    pos_encoding_output = positional_encoding(noise_level)
    print("Positional Encoding Output Shape:", pos_encoding_output.shape)

    ## Test TADE Layer
    tade_layer = TADELayer(activation_channels=64, conditioning_channels=64)
    tade_output = tade_layer(output_tensor, pos_encoding_output)
    print("TADE Layer Output Shape:", tade_output.shape)

    ## This works
    
    