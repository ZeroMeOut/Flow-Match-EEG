import torch 
import torch.nn as nn
from math import log as ln


## Thought process
## The input to the model is xt, t, and cond
## The model predicts the vector 
## xt goes into an embedding layer, maybe a transformer encoder layer
## t goes into a positional encoding layer
## cond goes into a transformer encoder layer
## Probably concatinate all of these inputs to one tensor

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8):
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

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class DenoisingModelUnit3(nn.Module):
    def __init__(self, channels, dim, num_resblocks=4):
        super().__init__()
        self.embed_dim = dim
        self.in_out_channels = channels

        # Embedding layers
        self.input_embedding = nn.Conv1d(channels, dim, kernel_size=1)
        self.cond_embedding = nn.Conv1d(channels, dim, kernel_size=1)
        self.time_embedding = PositionalEncoding(dim)

        # Encoder
        self.encoder = EncoderLayer(embed_dim=dim)

        # Residual blocks
        self.resblocks_1 = nn.ModuleList([ResBlock1D(dim) for _ in range(num_resblocks)])
        self.resblocks_2 = nn.ModuleList([ResBlock1D(dim) for _ in range(num_resblocks)])
        self.resblocks_3 = nn.ModuleList([ResBlock1D(dim) for _ in range(num_resblocks)])

        # Convolutional layers
        self.first_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.second_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.final_conv = nn.Conv1d(dim, channels, kernel_size=1)

    def forward(self, x, t, cond):
        # Embedding
        x_emb = self.input_embedding(x)
        cond_emb = self.cond_embedding(cond)
        time_emb = self.time_embedding(t)

        # Fuse embeddings
        fused = x_emb + cond_emb + time_emb

        # Encode
        encoded = self.encoder(fused)
        encoded_2 = encoded.clone()

        # First residual block path
        for block in self.resblocks_1:
            encoded = block(encoded)
        encoded = self.first_conv(encoded)

        # Second residual block path
        for block in self.resblocks_2:
            encoded_2 = block(encoded_2)
        encoded_2 = self.second_conv(encoded_2)

        # Combine and final residual blocks
        out = encoded * encoded_2
        for block in self.resblocks_3:
            out = block(out)
        out = self.final_conv(out)
        return out

## Testing
if __name__ == "__main__":
    model = DenoisingModelUnit3(1, 64)
    x = torch.randn(64, 1, 1500)
    t = torch.randn(64, 1)
    cond = torch.randn(64, 1, 1500)
    out = model(x, t, cond)
    print(out.shape)
            
        