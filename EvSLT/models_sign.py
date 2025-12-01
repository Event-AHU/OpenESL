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

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypergraph import build_H_and_G_from_tokens
from hypergraph import HGNN
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from attn_layer import GSF_model

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
    
    resnet_model = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    # model.fc = nn.Identity()

    return resnet_model

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

class mamba_model(nn.Module):
    def __init__(self, depth=2):
        super(mamba_model, self).__init__()
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.mamba_block = create_block(d_model=512, depth=depth, drop_path=0.)

    def forward(self, x):
        
        hidden_states = x
        
        for block in self.mamba_block:
            hidden_states, res_x = block(hidden_states)

        return hidden_states

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
        self.mamba_model = mamba_model(depth=2)
        self.avgpooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.pos_embed_start = nn.Parameter(torch.zeros(1, 49*8+1, 512))
        self.pos_embed = nn.Parameter(torch.zeros(1, 49*8+2, 512))
        
        trunc_normal_(self.pos_embed_start, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
        self.projector = nn.Linear(512, embed_dim)
        
        self.build_G = build_H_and_G_from_tokens
        self.hypergraph_encoder = HGNN(in_ch=512, n_class=1024, n_hid=1024)
        
        self.GSF_model = GSF_model(depth=1, dim=embed_dim, num_heads=8)
        
        self.mbart = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)
        self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=embed_dim)
        self.embed_scale = 1.0
            
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def visual_forward(self, src_input):
        group_size = 8
        attention_mask = src_input['attention_mask']
    
        out_visual = self.resnet(src_input['input_ids'].cuda(), src_input['src_length_batch']) # torch.Size([1152, 3, 224, 224]) → torch.Size([8, 128, 512, 7, 7])
        B, N, C, h, w = out_visual.shape
        
        global_feat_list = []
        video_clip_list = [out_visual[:, i:i+group_size, :, :, :] for i in range(0, N, group_size)]
        for video_clip in video_clip_list:
            global_token = self.avgpooling(video_clip).mean(1).unsqueeze(1).flatten(2)
            if len(global_feat_list) > 0:
                global_token = torch.cat((global_feat_list[-1], global_token), dim=1)
                
            video_clip_feat = video_clip.transpose(1,2).flatten(2).transpose(1,2)
            mamba_in_feat = torch.cat((global_token, video_clip_feat), dim=1)
            if len(global_feat_list) > 0:
                mamba_in_feat = mamba_in_feat + self.pos_embed
            else:
                mamba_in_feat = mamba_in_feat + self.pos_embed_start
                
            mamba_out_feat = self.mamba_model(mamba_in_feat)
            if len(global_feat_list) > 0:
                clip_out_feat = mamba_out_feat[:,2:,:]
                global_feat = mamba_out_feat[:,1:2,:]
            else:
                clip_out_feat = mamba_out_feat[:,1:,:]
                global_feat = mamba_out_feat[:,:1,:]
                
            global_feat_list.append(global_feat)
        
        spatial_feat = self.projector(clip_out_feat)
        global_feats = torch.cat((global_feat_list), dim=1)
        
        _, G = self.build_G(global_feats)
        hg_feat = self.hypergraph_encoder(global_feats, G)
        
        temporal_feat = self.avgpooling(out_visual).flatten(2)
        temporal_feat = self.conv_1d(temporal_feat)

        frames_feature = self.GSF_model(spatial_feat, hg_feat, temporal_feat)

        frames_feature = frames_feature[:,:src_input['new_src_length_batch'][0]]
        
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

