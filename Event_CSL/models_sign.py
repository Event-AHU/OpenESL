import torch
import torch.nn as nn
import numpy as np
import torchvision


from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig

from torch.nn.utils.rnn import pad_sequence
# global definition
from definition import *

from models_mamba import create_block


import math
from einops import rearrange, repeat

def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self, name="resnet18"):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name=name)

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x

class time_mamba(nn.Module):
    def __init__(self):
        super(time_mamba, self).__init__()

        self.mamba_block = create_block(d_model=512)

    def forward(self, x):

        hidden_x, res_x = self.mamba_block(x)

        return hidden_x

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)

class V_encoder(nn.Module):
    def __init__(self, emb_size, feature_size):
        super(V_encoder, self).__init__()

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, src:torch.Tensor):
      
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src

class SignModel(nn.Module):
    def __init__(self, config, args, embed_dim=1024):
        super(SignModel, self).__init__()
        self.args = args
        self.resnet = resnet(name="resnet18")
        self.time_mamba = time_mamba()
        self.mbart = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)
        self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=embed_dim)
        self.embed_scale = 1.0

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def visual_forward(self, src_input):

        attention_mask = src_input['attention_mask']

        out_visual = self.resnet(src_input['input_ids'].cuda(), src_input['src_length_batch'])
        out_visual = self.time_mamba(out_visual)
        frames_feature = self.conv_1d(out_visual)
        inputs_embeds = self.sign_emb(frames_feature) 
        inputs_embeds = self.embed_scale * inputs_embeds 
        return inputs_embeds, attention_mask

    def forward(self, src_input, tgt_input):
        inputs_embeds, attention_mask = self.visual_forward(src_input)
        out = self.mbart(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),
                    labels = tgt_input['input_ids'].cuda(),
                    decoder_attention_mask = tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )

        return out['logits']

    def generate(self,src_input, max_new_tokens, num_beams, decoder_start_token_id):

        inputs_embeds, attention_mask = self.visual_forward(src_input)
        out = self.mbart.generate(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),max_new_tokens=max_new_tokens,num_beams = num_beams,
                                decoder_start_token_id=decoder_start_token_id
                            )
        return out

