import torch
import torch.nn as nn
import torchvision.models as models
########---------------------------####
import copy
import math
import torch.nn.functional as F
########---------------------------####

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        #model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        model = models.resnet101(pretrained=False)
        pre = torch.load("/home/liyaw22/R2GenCMN/modules/resnet101-5d3b4d8f.pth")
        model.load_state_dict(pre)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        ########################---------------------------#########################
        embed_dim = 2048
        dropout = 0.1
        self.MultiHeadedAttention=MultiHeadedAttention(8, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        # 构造FeedForward层
        ffn_embed_dim = int(embed_dim * 4)
        self.ff_layer = FeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            relu_dropout=dropout
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        ########################---------------------------#########################


    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        ########################---------------------------#########################
        #print("grid_feture:   ",patch_feats.shape)
        #print("global_feature:   ",avg_feats.shape)
        gx = avg_feats.unsqueeze(1)
        
        global_feature = self.MultiHeadedAttention(gx,gx,gx)
        grid_feture = self.MultiHeadedAttention(patch_feats,patch_feats,patch_feats)
        
        x = torch.cat([grid_feture, global_feature], 1) 
        x = self.dropout(x)
        x = self.layer_norm1(x)

        # FeedForward及残差
        short_cut = x
        x = self.ff_layer(x)
        # dropout 残差 LayerNorm在此加入
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)
        patch_feats = x[:, :-1, :]
        avg_feats = x[:, -1, :]
        ########################---------------------------#########################
        return patch_feats, avg_feats
        
        

# 不包含残差连接和LayerNorm
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()  # ReLU / GELU / CELU
        #self.act = nn.GELU()
        
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x        
        
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])        

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn    
        
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)
