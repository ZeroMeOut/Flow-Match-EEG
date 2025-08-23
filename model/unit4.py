import torch 
import torch.nn as nn
from math import log as ln


## Thought process
## The input to the model is x
## The model predicts the vector 
## x goes into an embedding layer, maybe a transformer encoder layer
## Then output, profit trust

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
        x = x.permute(0, 2, 1)
        x = self.TransformerEncoderLayer(x)
        x = x.permute(0, 2, 1)
        return x

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, seq_len=1500):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        
        self.gn1 = nn.GroupNorm(min(32, channels//4), channels)
        self.gn2 = nn.GroupNorm(min(32, channels//4), channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class DenoisingModelUnit4(nn.Module):
    def __init__(self, channels, dim, num_resblocks=4, seq_len=1500):
        super(DenoisingModelUnit4, self).__init__()

        self.embed_dim = dim
        self.in_out_channels = channels
        self.seq_len = seq_len

        self.input_embedding = nn.Conv1d(self.in_out_channels, self.embed_dim, kernel_size=1)
        
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
    
    def forward(self, noisy_eeg):
        ## Input embedding
        x_embedded = self.input_embedding(noisy_eeg)
        x_embedded = self.input_norm(x_embedded)
        
        ## Encoder
        encoded_x = self.encoder(x_embedded)
        
        ## Residual processing
        out = encoded_x
        for block in self.resblocks:
            out = block(out)
        
        ## Final output
        output = self.final_conv(out)
        return output

## Testing
if __name__ == "__main__":
    model = DenoisingModelUnit4(1, 64)
    x = torch.randn(64, 1, 1500)
    out = model(x)
    print(out.shape)
            
        