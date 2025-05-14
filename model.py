import math
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import TransformerEncoder
from sklearn.metrics import roc_auc_score, f1_score

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters:
        -----------
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GRUModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 input_dim, embed_dim, nhead, d_hid, nlayers, lr):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.lr = lr

        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])
        
        self.transformer = TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=self.nhead, dim_feedforward=d_hid, batch_first=False),
            num_layers=nlayers
        )
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.fc_for_cls_token = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, patch, mask):
        batch_size, seq_len, _ = patch.size()
        src = self.embedding(patch) #* math.sqrt(self.embed_dim)

        x = x.permute(1, 0, 2)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # create mask
        attn_mask = torch.full((batch_size, seq_len + 1, seq_len + 1), float('-inf'), device=x.device) # (batch_size, 97 + CLS, 97 + CLS)
        attn_mask[:, 0, :].zero_() # unmasked cls token
        attn_mask[:, :, 0].zero_() # unmasked cls token

        outputs = []

        cls_tokens = self.cls_token.expand(src.size(0), 1, -1).to(src.device)  # (batch_size, 1, d_model)
        src = torch.cat((cls_tokens, src), dim=1) # (batch_size, seq len + CLS, d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)

        for t in range(seq_len):
            src_mask = self.update_mask(attn_mask, mask, t)
            new_src_mask = src_mask.copy() #self.prepare_data_before_transformer(src, src_mask)
            trans_output = self.transformer(src, new_src_mask)
            trans_output = trans_output.permute(1, 0, 2)
            cls_token = self.fc_for_cls_token(trans_output[:, 0, :])
            cur_x = torch.cat([x[t], cls_token], dim=1)
            
            h_new = [None] * self.num_layers
            h_new[0] = self.gru_cells[0](cur_x, h[0])
            for i in range(1, self.num_layers):
                h_new[i] = self.gru_cells[i](h_new[i-1], h[i])
            h = torch.stack(h_new)
            
            outputs.append(h[-1].unsqueeze(0))
        
        out = torch.cat(outputs, dim=0)
        out = out.transpose(0, 1)
        out = self.fc(out)
        out = self.softmax(out)
        out = self.weight_pool(out)
        return out[:, 1].to(torch.float32)

    def weight_pool(self, output):
        weights = np.arange(0, output.size(1))
        weights = weights / np.sum(weights)
        weights = np.repeat(np.expand_dims(weights, axis=0), output.size(0), axis=0)
        weights = np.repeat(np.expand_dims(weights, axis=2), 2, axis=2)
        weights = torch.tensor(weights, device=output.device)
        results = torch.mul(output, weights)
        results = torch.sum(results, axis=1)
        return results
    
    # def prepare_data_before_transformer(self, src, src_mask):
    #     device = src.device
    #     batch_size, seq_len = src_mask.shape[0], src_mask.shape[1]
    #     new_src_mask = torch.full((batch_size, seq_len + 1, seq_len + 1), float('-inf'), dtype=src_mask.dtype, device=device)
    #     new_src_mask[:, 0, :].zero_()#new_src_mask[:, :, 0].zero_()
    #     new_src_mask[:, 1:, 1:] = src_mask
    #     expanded_mask = self.expand_mask_for_heads(new_src_mask)
    #     return expanded_mask
    
    def expand_mask_for_heads(self, src_mask):
        """
        Expand the attention mask to fit the multi-head attention mechanism.

        Parameters:
        -----------
            src_mask -- the original source mask with shape (batch_size, seq_len, seq_len)

        Returns:
        -----------
            expanded_mask -- the expanded mask with shape (batch_size * nhead, seq_len, seq_len)
        """
        batch_size, seq_len, _ = src_mask.size()
        expanded_mask = src_mask.repeat(1, self.nhead, 1).reshape(batch_size * self.nhead, seq_len, seq_len)
        return expanded_mask

    def update_mask(self, attn_mask, mask, t):
        important_indices = mask[:, : t+1]
        for batch_idx, indices in enumerate(important_indices):
            for idx in indices:
                attn_mask[batch_idx, :, idx] = 0
        return attn_mask
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, patch, mask, y = batch
        predictions = self(x, patch.flatten(start_dim=2), mask)
        loss = nn.functional.binary_cross_entropy(predictions, y.float())
        if y.unique().numel() == 2:
            try:
                roc_auc = roc_auc_score(y.cpu(), predictions.cpu().detach().numpy())
                self.log('train_roc_auc', roc_auc)

                f1 = f1_score(y.cpu(), predictions.cpu().round().detach().numpy())
                self.log('train_f1', f1)
            except ValueError:
                pass
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, patch, mask, y = batch
        predictions = self(x, patch.flatten(start_dim=2), mask)
        loss = nn.functional.binary_cross_entropy(predictions, y.float())
        
        if y.unique().numel() == 2:
            try:
                roc_auc = roc_auc_score(y.cpu(), predictions.cpu().detach().numpy())
                self.log('val_roc_auc', roc_auc)

                f1 = f1_score(y.cpu(), predictions.cpu().round().detach().numpy())
                self.log('val_f1', f1)
            except ValueError:
                pass
    
        self.log('val_loss', loss)
        
        return {'val_loss': loss, 'val_f1': f1 if y.unique().numel() == 2 else None, 'val_roc_auc': roc_auc if y.unique().numel() == 2 else None}

