
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer

class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, condition_keys = ['race', 'sex'], num_classes = {'race':3, 'sex':2}, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(num_classes[key], emb_dim) for key in condition_keys
        })

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        """
        condition:[key: tensor(B,), ...]
        """
        cond_embs = [self.embeddings[k](condition[k]) for k in condition]
        c = torch.cat(cond_embs, dim=1)
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c



