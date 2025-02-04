import math

import torch
import torch.nn as nn

from mamba_ssm import Mamba, Mamba2
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig


def minmaxnormalize(x):
    # Min-max normalization over the last dimension (P), per sample (B, S)
    x_min = x.min(dim=-1, keepdim=True)[0]  # Shape: (B, S, 1)
    x_max = x.max(dim=-1, keepdim=True)[0]  # Shape: (B, S, 1)
    x = (x - x_min) / (x_max - x_min + 1e-6)
    return x
  

class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5, embedder=None, fs=40, minmax=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Output dimension can be set as needed

        # Specify layers where manifold mixup can be applied
        self.manifold_mixup_layers = ['lstm_output', 'context']
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.embedder = embedder
        self.fs = fs
        self.minmax = minmax

    def forward(self, x, mixup_layer=None, lam=None, index=None):
        # x shape: (batch_size, seq_length, input_dim)
        if self.minmax:
            x = minmaxnormalize(x)

        if self.embedder is not None:
            assert x.shape[-1] == self.fs, f'{x.shape=}'
            #
#            print(x.shape, type(x))
#            with torch.amp.autocast('cuda', enabled=True):
#                x = self.embedder.encode(x, apply_mask=False)[:,1:,:]
#                print(x.shape)
#                raise

            with torch.amp.autocast('cuda', enabled=True):
                batch_size, S, P = x.shape
                segment_length = 30
                n_segments = S // segment_length  # Make sure S is divisible by segment_length
                output_list = []

                # Split sequence into 30-length segments
                for i in range(n_segments):
                    start_idx = i * segment_length
                    end_idx = start_idx + segment_length

                    # Extract segment (B, segment_length, P)
                    segment = x[:, start_idx:end_idx, :]

                    # Process through embedder (B, segment_length, D)
                    encoded_segment = self.embedder.encode(segment, apply_mask=False)

                    # Take only the last token from each segment (B, 1, D)
                    last_token = encoded_segment[:, -1:, :]

                    output_list.append(last_token)

                # Concatenate all final tokens (B, n_segments, D)
                final_output = torch.cat(output_list, dim=1)
                x = final_output

        h_lstm, _ = self.lstm(x)  # (batch_size, seq_length, hidden_dim * 2)
        h_lstm = self.layer_norm(h_lstm)

        # Potential manifold mixup point after LSTM output
        if mixup_layer == 'lstm_output' and lam is not None and index is not None:
            h_lstm = lam * h_lstm + (1 - lam) * h_lstm[index, :, :]

        attn_weights = torch.softmax(self.attention(h_lstm), dim=1)  # (batch_size, seq_length, 1)
        context = torch.sum(attn_weights * h_lstm, dim=1)  # (batch_size, hidden_dim * 2)

        # Potential manifold mixup point after attention context
        if mixup_layer == 'context' and lam is not None and index is not None:
            context = lam * context + (1 - lam) * context[index, :]

        context = self.dropout(context)
        out = self.fc(context)  # (batch_size, output_dim)
        return out

class MambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5, embedder=None, fs=40):
        super().__init__()
        self.input_embedding_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Create a stack of Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,  # Model dimension
                d_state=16,          # SSM state expansion factor
                d_conv=4,            # Local convolution width
                expand=2,            # Block expansion factor
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # Output layer for binary classification

        # Specify layers where manifold mixup can be applied
        self.manifold_mixup_layers = ['mamba_output', 'context']
        self.embedder = embedder
        self.fs = fs

    def forward(self, x, mixup_layer=None, lam=None, index=None):
        # x shape: (batch_size, seq_length, input_dim)
        x = minmaxnormalize(x)
        if self.embedder is not None:
            assert x.shape[-1] == self.fs, f'{x.shape=}'
##            print(x.shape, type(x))
#            with autocast('cuda', enabled=True):
#                x = self.embedder.encode(x, apply_mask=False)
##            raise Exception(x.shape)
            with torch.amp.autocast('cuda', enabled=True):
                batch_size, S, P = x.shape
                segment_length = 30
                n_segments = S // segment_length  # Make sure S is divisible by segment_length
                output_list = []
                # Split sequence into 30-length segments
                for i in range(n_segments):
                    start_idx = i * segment_length
                    end_idx = start_idx + segment_length
                    # Extract segment (B, segment_length, P)
                    segment = x[:, start_idx:end_idx, :]
                    # Process through embedder (B, segment_length, D)
                    encoded_segment = self.embedder.encode(segment, apply_mask=False)
                    # Take only the last token from each segment (B, 1, D)
                    last_token = encoded_segment[:, -1:, :]
                    output_list.append(last_token)
                # Concatenate all final tokens (B, n_segments, D)
                final_output = torch.cat(output_list, dim=1)
                x = final_output

        # Project input to hidden_dim
        x = self.input_embedding_norm(x)
        x_proj = self.input_proj(x)  # (batch_size, seq_length, hidden_dim)

        h = x_proj
        for layer in self.mamba_layers:
            h = layer(h)  # (batch_size, seq_length, hidden_dim)

        # Potential manifold mixup point after Mamba output
        if mixup_layer == 'mamba_output' and lam is not None and index is not None:
            h = lam * h + (1 - lam) * h[index, :, :]

        # Pooling over the time dimension (e.g., mean pooling)
#        context = h.mean(dim=1)  # (batch_size, hidden_dim)
        context, _ = torch.max(h, dim=1)

        # Potential manifold mixup point after context pooling
        if mixup_layer == 'context' and lam is not None and index is not None:
            context = lam * context + (1 - lam) * context[index, :]

        context = self.dropout(context)
        out = self.fc(context)  # (batch_size, 1)
        return out


class XLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout=0.5,
                 mlstm_block=None, slstm_block=None, slstm_at=None, context_length=256,
                 embedder=None, fs=40, minmax=True):
        super().__init__()
        self.input_embedding_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Set default configurations for MLSTM and SLSTM blocks
        if mlstm_block is None:
            mlstm_block = {
                'mlstm': {
                    'conv1d_kernel_size': 4,
                    'qkv_proj_blocksize': 4,
                    'num_heads': 4
                }
            }
        if slstm_block is None:
            slstm_block = {
                'slstm': {
                    'backend': 'cuda',
                    'num_heads': 4,
                    'conv1d_kernel_size': 4,
                    'bias_init': 'powerlaw_blockdependent'
                },
                'feedforward': {
                    'proj_factor': 1.3,
                    'act_fn': 'gelu'
                }
            }
        if slstm_at is None:
            slstm_at = [1]

        # Build xLSTM configuration dictionary
        xlstm_cfg_dict = {
            'mlstm_block': mlstm_block,
            'slstm_block': slstm_block,
            'context_length': context_length,
            'num_blocks': num_blocks,
            'embedding_dim': hidden_dim,
            'slstm_at': slstm_at
        }

        # Convert to xLSTMBlockStackConfig using OmegaConf and dacite
        cfg = OmegaConf.create(xlstm_cfg_dict)
        cfg = from_dict(
            data_class=xLSTMBlockStackConfig,
            data=OmegaConf.to_container(cfg, resolve=True),
            config=DaciteConfig(strict=True)
        )
        self.xlstm_stack = xLSTMBlockStack(cfg)

        # Additional layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # Binary classification output

        # Embedding and preprocessing parameters
        self.embedder = embedder
        self.fs = fs
        self.minmax = minmax

        # Mixup layers
        self.manifold_mixup_layers = ['xlstm_output', 'context']

    def forward(self, x, mixup_layer=None, lam=None, index=None):
        # Handle packed sequences
        if isinstance(x, rnn.PackedSequence):
            x_packed = x
            x, lengths = rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x_packed = None
            lengths = None

        # Min-max normalization
        if self.minmax:
            x = minmaxnormalize(x)

        # Apply embedder if provided
        if self.embedder is not None:
            assert x.shape[-1] == self.fs, f"Input feature dim {x.shape[-1]} != embedder expected {self.fs}"
#            with torch.autocast(device_type='cuda', enabled=True):
#                x = self.embedder.encode(x, apply_mask=False)[:, 1:, :]
            with torch.amp.autocast('cuda', enabled=True):
                batch_size, S, P = x.shape
                segment_length = 30
                n_segments = S // segment_length  # Make sure S is divisible by segment_length
                output_list = []
                # Split sequence into 30-length segments
                for i in range(n_segments):
                    start_idx = i * segment_length
                    end_idx = start_idx + segment_length
                    # Extract segment (B, segment_length, P)
                    segment = x[:, start_idx:end_idx, :]
                    # Process through embedder (B, segment_length, D)
                    encoded_segment = self.embedder.encode(segment, apply_mask=False)
                    # Take only the last token from each segment (B, 1, D)
                    last_token = encoded_segment[:, -1:, :]
                    output_list.append(last_token)
                # Concatenate all final tokens (B, n_segments, D)
                final_output = torch.cat(output_list, dim=1)
                x = final_output

        # Input projection
        x = self.input_embedding_norm(x)
        x_proj = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)

        # Process through xLSTM blocks
        h = self.xlstm_stack(x_proj)
        h = self.layer_norm(h)

        # Manifold mixup after xLSTM output
        if mixup_layer == 'xlstm_output' and lam is not None and index is not None:
            h = lam * h + (1 - lam) * h[index, :, :]

        # Context pooling
        context, _ = torch.max(h, dim=1)

        # Mixup after context
        if mixup_layer == 'context' and lam is not None and index is not None:
            context = lam * context + (1 - lam) * context[index, :]

        # Final output
        context = self.dropout(context)
        out = self.fc(context)
        return out
