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
        self.time_proj = nn.Linear(dim, dim)
        
    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        
        encoding = self.time_proj(encoding)
        return encoding.unsqueeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, seq_len=1500):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        
        self.gn1 = nn.GroupNorm(min(32, channels//4), channels)
        self.gn2 = nn.GroupNorm(min(32, channels//4), channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)  # 
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.gn1(out)  
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.gn2(out)  
        
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class DenoisingModelUnit2(nn.Module):
    def __init__(self, channels, dim, num_resblocks=4, seq_len=1500):
        super(DenoisingModelUnit2, self).__init__()

        self.embed_dim = dim
        self.in_out_channels = channels
        self.seq_len = seq_len

        self.time_embedding = PositionalEncoding(self.embed_dim)
        self.input_embedding = nn.Conv1d(self.in_out_channels, self.embed_dim, kernel_size=1)
        self.cond_embedding = nn.Conv1d(self.in_out_channels, self.embed_dim, kernel_size=1)

        self.time_proj = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.cond_proj = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1)
        
        self.input_norm = nn.GroupNorm(min(32, self.embed_dim//4), self.embed_dim)
        
        self.encoder = EncoderLayer(embed_dim=self.embed_dim)

        resblocks = [ResBlock1D(self.embed_dim, seq_len=seq_len) for _ in range(num_resblocks)]
        self.resblocks = nn.ModuleList(resblocks)
        
        self.final_conv = nn.Conv1d(self.embed_dim, self.in_out_channels, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, t, cond):
        ## Input embeddings
        x_embedded = self.input_embedding(x)
        cond_embedded = self.cond_embedding(cond)

        time_emb = self.time_embedding(t)  # [batch, dim, 1]
        time_emb = time_emb.expand(-1, -1, x.size(-1))  # [batch, dim, seq_len]
        time_emb = self.time_proj(time_emb)
        
        ## Condition embedding projection
        cond_embedded = self.cond_proj(cond_embedded)
        
        ## FiLM-like conditioning
        fused_input = x_embedded * (1 + time_emb) + cond_embedded
        fused_input = self.input_norm(fused_input)
        
        ## Encoder
        encoded_x = self.encoder(fused_input)
        
        ## Residual processing
        out = encoded_x
        for block in self.resblocks:
            out = block(out)
        
        ## Final output
        output = self.final_conv(out)
        return output

## Testing
if __name__ == "__main__":
    model = DenoisingModelUnit2(1, 64)
    x = torch.randn(64, 1, 1500)
    t = torch.randn(64, 1)
    cond = torch.randn(64, 1, 1500)
    out = model(x, t, cond)
    print(out.shape)
            
        