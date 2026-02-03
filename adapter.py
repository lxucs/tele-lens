import json
import torch
import torch.nn as nn
import logging
from os.path import join, exists
import math
from transformers.trainer import WEIGHTS_NAME
from transformers import AutoModelForCausalLM
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class AdapterLayer(nn.Module):
    """ Bottleneck adapter. """
    def __init__(self, hidden_size: int, rank: int, alpha: float, dropout: float, activation=None, bias=False):
        super().__init__()
        self.hidden_size, self.r, self.alpha = hidden_size, rank, alpha
        self.scaling = self.alpha / self.r
        self.bias = bias

        self.proj_A = nn.Linear(self.hidden_size, self.r, bias=self.bias)
        self.proj_B = nn.Linear(self.r, self.hidden_size, bias=self.bias)
        self.activation = self._get_activation(activation) if activation else nn.Identity()
        self.dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.proj_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_B.weight, a=math.sqrt(5))
        if self.bias:
            nn.init.zeros_(self.proj_A.bias)
            nn.init.zeros_(self.proj_B.bias)

    @classmethod
    def _get_activation(cls, name):
        name = (name or '').lower()
        return nn.ReLU() if name == 'relu' else nn.GELU() if name == 'gelu' else nn.SiLU() if name in ('silu', 'swish') else nn.Tanh() if name == 'tanh' else None

    def forward(self, hidden):
        return self.proj_B(self.activation(self.proj_A(self.dropout(hidden)))) * self.scaling

    @property
    def device(self):
        return self.proj_A.weight.device


class TransformModel(nn.Module):

    @classmethod
    def config_path(cls, model_path):
        return join(model_path, 'model_config.json')

    @classmethod
    def state_path(cls, model_path):
        return join(model_path, WEIGHTS_NAME)

    def __init__(self, model_path: str=None, llm_path: str=None, candidate_vocab_ids: list=None, cond_size: int=None,
                 rank: int=None, alpha: float=None, dropout: float=None, activation: str=None):
        super().__init__()

        # Set config
        conf = {}
        if model_path and exists(self.config_path(model_path)):
            with open(self.config_path(model_path), 'r') as f:
                conf = json.load(f)
            logger.info(f'Loaded config from {self.config_path(model_path)}: {conf}')
        passed_conf = {'llm_path': llm_path, 'candidate_vocab_ids': candidate_vocab_ids, 'cond_size': cond_size,
                       'rank': rank, 'alpha': alpha, 'dropout': dropout, 'activation': activation}
        for k, v in passed_conf.items():
            if k not in conf or (v is not None and v != conf[k]):
                conf[k] = v
                print(f'Override config {k}: {v}')

        # Set LM head (assuming w/o bias)
        if not conf.get('hidden_size', None):
            # Load from LLM
            lm_head_weight = AutoModelForCausalLM.from_pretrained(conf['llm_path'], trust_remote_code=True).lm_head.weight.data.detach()
            if conf['candidate_vocab_ids']:
                lm_head_weight = lm_head_weight[torch.tensor(conf['candidate_vocab_ids'], dtype=torch.long), :].contiguous()
            else:
                logger.info(f'Using all vocabs (candidate_vocab_ids is empty)')

            conf['vocab_size'], conf['hidden_size'] = lm_head_weight.size()  # Torch weight convention
            self.register_buffer('lm_head_weight', lm_head_weight, persistent=True)
            logger.info(f'Loaded LM head weight ({conf['hidden_size']} x {conf['vocab_size']}) from {conf['llm_path']}')
        else:
            lm_head_weight = nn.Linear(conf['hidden_size'], conf['vocab_size'], bias=False).weight.data.detach()
            self.register_buffer('lm_head_weight', lm_head_weight, persistent=True)

        # Set param
        self.adapter = AdapterLayer(conf['hidden_size'], conf['rank'], conf['alpha'], conf['dropout'], conf['activation']) if conf['rank'] else None
        self.embed_cond = nn.Embedding(conf['cond_size'] + 1, conf['hidden_size']) if conf['cond_size'] else None
        if model_path and exists(self.state_path(model_path)):
            self.load_state_dict(torch.load(self.state_path(model_path), map_location='cpu', weights_only=True))
            logger.info(f'Loaded model state from {self.state_path(model_path)}')
        else:
            logger.info(f'Initialized model state')

        self.conf = conf

    @property
    def device(self):
        return self.lm_head_weight.device

    def save(self, model_path):
        # Save config
        with open(self.config_path(model_path), 'w') as f:
            json.dump(self.conf, f, indent=2, ensure_ascii=False)
        # Save state
        torch.save(self.state_dict(), self.state_path(model_path))

    def forward_without_adapter(self, feat_hidden, **kwargs):
        logits = F.linear(feat_hidden, self.lm_head_weight)
        return logits

    def forward(self, feat_hidden, group_sizes=None, conds=None, labels=None, **kwargs):
        if conds is not None:
            feat_hidden = feat_hidden.repeat_interleave(group_sizes, dim=0)
            feat_hidden += self.embed_cond(conds)

        hidden = self.adapter(feat_hidden)
        logits = F.linear(hidden, self.lm_head_weight)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels.view(-1))
            return loss, logits
        return logits
