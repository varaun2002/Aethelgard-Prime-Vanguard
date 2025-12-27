import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class VolatilityEncoder(nn.Module):
    """
    Encodes static or slowly changing volatility features (ADX, CHOP, BB Width, ATR Rank)
    into a dense embedding to modulate the main backbone.
    """
    def __init__(self, input_dim, embed_dim=64):
        super(VolatilityEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh() # Tanh to output modulation weights roughly -1 to 1
        )
        
    def forward(self, x):
        # x: [batch_size, vol_feature_dim] (Taken from last time step)
        return self.net(x)

class CrossAttentionFusion(nn.Module):
    """
    Fuses Temporal Features (Backbone) with Regime/Volatility Context using Attention.
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.mixed_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism
        self.gate = nn.Linear(d_model * 2, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, context):
        # query: [batch_size, 1, d_model] (Usually the Regime Embedding or Last Timestep)
        # context: [batch_size, seq_len, d_model] (The full temporal sequence from Transformer/GRU)
        
        attn_out, _ = self.mixed_attn(query, context, context)
        # attn_out: [batch_size, 1, d_model]
        
        # Residual connection + Norm
        fused = self.norm(query + self.dropout(attn_out))
        return fused.squeeze(1) # [batch_size, d_model]

class CryptoModelV3(nn.Module):
    def __init__(self, input_dim=40, vol_dim=0, d_model=64, nhead=4, num_layers=1, dropout=0.4):
        """
        v2.3 Architecture with Cross-Attention Fusion and Confidence Calibration.
        
        Args:
            vol_dim: Number of features to treat as "Volatility Context" (ADX, CHOP, etc.)
                     If 0, it behaves like v2.2 but with better heads.
        """
        super(CryptoModelV3, self).__init__()
        
        self.d_model = d_model
        
        # --- Backbones ---
        # 1. Temporal Backbone (GRU)
        self.gru = nn.GRU(input_dim, d_model, num_layers, batch_first=True, dropout=dropout)
        
        # 2. Context Backbone (Transformer)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 3. Volatility Encoder (New in v2.3)
        self.use_vol_encoder = vol_dim > 0
        if self.use_vol_encoder:
            self.vol_encoder = VolatilityEncoder(vol_dim, d_model)
            self.fusion_layer = CrossAttentionFusion(d_model, nhead, dropout)
        else:
            self.fusion_linear = nn.Linear(d_model * 2, d_model) # Fallback to v2.2 style
            
        self.fusion_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # --- Heads ---
        # 1. Regime Head (4 Classes: Trend UP, Trend DOWN, Range Quiet, Range Noisy)
        self.head_regime = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4) 
        )
        
        # 2. Direction Head (Probabilities + Confidence + Uncertainty)
        # Note: Uncertainty is estimated via MC Dropout, not a direct output, 
        # but we add a specific variance head just in case we want Aleatoric uncertainty.
        self.head_direction = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout), # Critical for MC Dropout
            nn.Linear(64, 3) # UP, DOWN, FLAT logits
        )
        
        # 3. Return Head (Huber-ready)
        self.head_return = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, vol_features=None):
        # x: [batch_size, seq_len, input_dim]
        # vol_features: [batch_size, vol_dim] (Optional, static context)
        
        # --- Transformer Path ---
        x_t = self.embedding(x).permute(1, 0, 2) # [seq, batch, dim]
        x_t = self.pos_encoder(x_t)
        x_t = x_t.permute(1, 0, 2) # [batch, seq, dim]
        trans_out = self.transformer_encoder(x_t)
        
        # --- GRU Path ---
        gru_out, _ = self.gru(x)
        
        # --- Fusion ---
        if self.use_vol_encoder and vol_features is not None:
            # Encode Volatility Context
            vol_embed = self.vol_encoder(vol_features).unsqueeze(1) # [batch, 1, d_model]
            
            # Cross Attention: Volatility "queries" the Time Series "context"
            # We treat the Transformer output as the rich context
            fused = self.fusion_layer(vol_embed, trans_out) # [batch, d_model]
            
            # We supplement this with the latest GRU state for immediate trend
            gru_last = gru_out[:, -1, :]
            fused = fused + gru_last # Skip connection style
            
        else:
            # Fallback (Concat)
            trans_last = trans_out[:, -1, :]
            gru_last = gru_out[:, -1, :]
            combined = torch.cat([gru_last, trans_last], dim=1)
            fused = self.fusion_linear(combined)
            
        # Global Norm & Dropout
        fused = self.fusion_norm(fused)
        fused = torch.relu(fused)
        fused = self.dropout_layer(fused)
        
        # --- Multi-Task Outputs ---
        return {
            'regime_logits': self.head_regime(fused),
            'direction_logits': self.head_direction(fused),
            'predicted_return': self.head_return(fused)
        }
    
    def mc_dropout_predict(self, x, vol_features=None, n_samples=10):
        """
        Runs inference multiple times with dropout enabled to estimate uncertainty.
        """
        self.train() # Enable Dropout
        
        dir_probs_list = []
        return_preds_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, vol_features)
                
                # Direction Probabilities
                d_logits = out['direction_logits']
                d_probs = F.softmax(d_logits, dim=1)
                dir_probs_list.append(d_probs.unsqueeze(0))
                
                # Return Predictions
                return_preds_list.append(out['predicted_return'].unsqueeze(0))
        
        # Stack results
        # [n_samples, batch, 3]
        all_dir_probs = torch.cat(dir_probs_list, dim=0) 
        # [n_samples, batch, 1]
        all_returns = torch.cat(return_preds_list, dim=0)
        
        # Calculate Mean (Prediction)
        mean_dir_probs = all_dir_probs.mean(dim=0)
        mean_return = all_returns.mean(dim=0)
        
        # Calculate Variance (Uncertainty)
        # Epistemic Uncertainty = Variance of the softmax probabilities
        # A high variance means the model oscillates between classes heavily
        dir_variance = all_dir_probs.var(dim=0).sum(dim=1) # Sum variance across classes
        return_variance = all_returns.var(dim=0)
        
        self.eval() # Reset to Eval mode
        
        return {
            'direction_probs': mean_dir_probs,
            'direction_uncertainty': dir_variance, # Scalar confidence penalty
            'predicted_return': mean_return,
            'return_uncertainty': return_variance
        }
