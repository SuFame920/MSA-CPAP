import torch
import torch.nn.functional as F
import math
import copy
from random import randint
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel,BertConfig
from .transformer import TransformerEncoder,TransformerEncoderLayer
from torch.nn import TransformerEncoder as TransEncoder
from torch.nn import TransformerEncoderLayer as TransEncoderLayer
from transformers import RobertaConfig, RobertaModel
from typing import Optional, Dict, Tuple
import numpy as np
from utils.masked_mean_pooling import masked_mean_pooling

def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens


# √
class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """

    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        self.hp = hp
        # Pretrained text encoders are downloaded from the HuggingFace Hub on first use
        # (cached under ~/.cache/huggingface). To use local weights instead, set the
        # environment variable HF_HUB_OFFLINE=1 and point these ids to a local directory.
        if self.hp.text_encoder == 'bert':
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        elif self.hp.text_encoder == 'roberta':
            roberta_config = RobertaConfig.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', output_hidden_states=True,
                                                        output_attentions=True)
            self.roberta_model = RobertaModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', config=roberta_config)

    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
        if self.hp.text_encoder == 'bert':
            bert_output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask)
        elif self.hp.text_encoder == 'roberta':
            bert_output = self.roberta_model(input_ids=bert_sent, attention_mask=bert_sent_mask)

        bert_output = bert_output[0]
        return bert_output


# √
class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text', output_size=None):
        self.n_class=n_class
        self.output_size=output_size
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 2*hidden_size)
        self.linear_3 = nn.Linear(2*hidden_size, hidden_size)
        if output_size is not None:
            self.output_linear=nn.Linear(hidden_size,output_size)
        if n_class is not None:
            self.linear_4=nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_2 = self.drop(y_2)
        y_3 = torch.tanh(self.linear_3(y_2))
        if self.n_class is not None:
            y_4 = self.linear_4(y_3)
            return y_3, y_4
        elif self.output_size is not None:
            y_4 = self.output_linear(y_3)
            return y_4


# √
class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100, dropout=0.1, n_class=1):
        super(SumFusion, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(output_dim, 2 * output_dim)
        self.linear_3 = nn.Linear(2 * output_dim, output_dim)
        self.linear_4 = nn.Linear(output_dim, n_class)

    def forward(self, x, y):
        sum_result = x + y
        dropped = self.drop(sum_result)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_2 = self.drop(y_2)
        y_3 = torch.tanh(self.linear_3(y_2))
        preds = self.linear_4(y_3)
        return y_3, preds


# √
class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100, dropout=0.1, n_class=1):
        super(ConcatFusion, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(input_dim*2, output_dim)
        self.linear_2 = nn.Linear(output_dim, 2*output_dim)
        self.linear_3 = nn.Linear(2*output_dim, output_dim)
        self.fc_out = nn.Linear(input_dim, n_class)

    def forward(self, x, y):
        modal_cat = torch.cat((x, y), dim=1)
        dropped = self.drop(modal_cat)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_2 = self.drop(y_2)
        y_3 = torch.tanh(self.linear_3(y_2))
        preds = self.linear_4(y_3)
        return y_3, preds


# √
class FusionTrans(nn.Module):
    def __init__(self, hp, n_class):
        super(FusionTrans, self).__init__()
        self.hp = hp
        self.d_l, self.d_a, self.d_v = 30, 30, 30

        # 各种超参
        self.num_heads = hp.num_heads
        self.layers = hp.layers
        self.attn_dropout = hp.attn_dropout
        self.attn_dropout_a = hp.attn_dropout_a
        self.attn_dropout_v = hp.attn_dropout_v
        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout
        self.out_dropout = hp.out_dropout
        self.d_prjh = hp.d_prjh
        self.embed_dropout = hp.embed_dropout
        self.attn_mask = hp.attn_mask
        self.n_lv = hp.n_tv
        self.n_la = hp.n_ta

        # 1. 通道对齐 (Conv1d kernel=1 == 逐位置线性投影)
        self.proj_l = nn.Conv1d(hp.d_tin, self.d_l, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(hp.d_ain, self.d_a, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(hp.d_vin, self.d_v, kernel_size=1, bias=False)

        # 2. 跨模态 Transformer（具体实现假设你已有 TransformerEncoder）
        self.trans_l_with_a = self.get_network('la')
        self.trans_l_with_v = self.get_network('lv')

        # 3. 投影/分类头
        self.proj1 = nn.Linear(self.d_l, self.d_l)
        self.proj2 = nn.Linear(self.d_l, self.d_l)
        self.proj3 = nn.Linear(self.d_l, self.d_prjh)
        self.out_layer = nn.Linear(self.d_l, n_class)

    def get_network(self, self_type='l', layers=-1):
        """根据模态选择 TransformerEncoder 超参"""
        if self_type in ['l', 'al', 'vl', 'lla', 'llv']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )

    #########################################
    # forward：干净版本，全程保留序列，最后池化
    #########################################
    def forward(self, t, a, v, t_mask=None, a_mask=None, v_mask=None):
        """
        输入:
          t: [Tt, B, Dt]  文本序列
          a: [Ta, B, Da]  声学序列
          v: [Tv, B, Dv]  视觉序列
          *_mask: [B, T*]  1=PAD, 0=有效
        """

        # 1) 通道投影 (Conv1d, kernel=1)
        text = self.proj_l(t.permute(1, 2, 0)).permute(2, 0, 1)      # [Tt,B,30]
        acoustic = self.proj_a(a.permute(1, 2, 0)).permute(2, 0, 1)  # [Ta,B,30]
        visual   = self.proj_v(v.permute(1, 2, 0)).permute(2, 0, 1)  # [Tv,B,30]

        # 2) 第一次跨模态交互 (以 text 为 Query，融合 a/v)
        l_with_a = self.trans_l_with_a(text, acoustic, acoustic, q_mask=t_mask, kv_mask=a_mask)  # [Tt,B,30]
        l_with_v = self.trans_l_with_v(text, visual, visual, q_mask=t_mask, kv_mask=v_mask)      # [Tt,B,30]

        # 3) 多轮交互 (仍然保持序列形状)
        for i in range(max(self.n_la, self.n_lv)):
            if i < self.n_la:
                l_with_aa = self.trans_l_with_a(text, l_with_a, l_with_a, q_mask=t_mask, kv_mask=t_mask)
                l_with_av = self.trans_l_with_a(text, l_with_v, l_with_v, q_mask=t_mask, kv_mask=t_mask)
                l_with_a = 0.5 * (l_with_aa + l_with_av)  # [Tt,B,30]

            if i < self.n_lv:
                l_with_vv = self.trans_l_with_v(text, l_with_v, l_with_v, q_mask=t_mask, kv_mask=t_mask)
                l_with_va = self.trans_l_with_v(text, l_with_a, l_with_a, q_mask=t_mask, kv_mask=t_mask)
                l_with_v = 0.5 * (l_with_vv + l_with_va)  # [Tt,B,30]

        # 4) 融合分支 (仍然保持序列)
        if min(self.n_la, self.n_lv) > 0:
            seq = 0.5 * (l_with_a + l_with_v)     # [Tt,B,30]
        elif self.n_la == 0 and self.n_lv != 0:
            seq = l_with_v                        # [Tt,B,30]
        elif self.n_la != 0 and self.n_lv == 0:
            seq = l_with_a                        # [Tt,B,30]
        else:
            seq = 0.5 * (l_with_a + l_with_v)     # [Tt,B,30]

        # 5) 句级池化 (mask-aware mean pooling)
        last_hs = masked_mean_pooling(seq, t_mask)   # [B,30]

        # 6) 残差 MLP 块
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)),
                      p=self.out_dropout, training=self.training)
        )  # [B,30]
        last_hs_proj = last_hs_proj + last_hs        # 残差加回 [B,30]

        # 7) 双头输出
        feat = self.proj3(last_hs)                   # [B,d_prjh] 表征向量
        output = self.out_layer(last_hs_proj)        # [B,n_class] 分类/回归结果

        return feat, output


# √
class CrossAttention(nn.Module):
    def __init__(self, hp, d_modal1, d_modal2, d_model, nhead, dim_feedforward, dropout, num_layers=6):
        super(CrossAttention, self).__init__()
        self.hp = hp
        self.d_modal1 = d_modal1
        self.d_modal2 = d_modal2
        self.num_heads = nhead
        self.d_model = d_model
        self.proj_modal1 = nn.Conv1d(self.d_modal1, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_modal2 = nn.Conv1d(self.d_modal2, self.d_model, kernel_size=1, padding=0, bias=False)
        self.layers = num_layers
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.output_linear = nn.Linear(dim_feedforward, self.d_model)

        self.attn_dropout = dropout
        self.relu_dropout = self.hp.relu_dropout
        self.res_dropout = self.hp.res_dropout
        self.embed_dropout = self.hp.embed_dropout
        self.attn_mask = self.hp.attn_mask

        self.net = self.get_network()

    def get_network(self, layers=-1):
        return TransformerEncoder(embed_dim=self.d_model, 
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, input_modal1, input_modal2, Tmask=None, Amask=None, Vmask=None, whether_add_position=True, counterfactual_attention_type=None):
        """
        传入格式：(seq_len, batch_size, emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([815, 32, 20])
        """
        modal1 = self.proj_modal1(input_modal1.permute(1, 2, 0))
        modal2 = self.proj_modal2(input_modal2.permute(1, 2, 0))
        modal1 = modal1.permute(2, 0, 1)
        modal2 = modal2.permute(2, 0, 1)

        if self.hp.d_tin == self.d_modal1 and self.hp.d_ain == self.d_modal2:
            encoded= self.net(modal1, modal2, modal2,Tmask,Amask,whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_tin == self.d_modal1 and self.hp.d_vin == self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Tmask, Vmask,whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_ain == self.d_modal1 and self.hp.d_tin == self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Amask, Tmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_ain == self.d_modal1 and self.hp.d_vin == self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Amask, Vmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_vin == self.d_modal1 and self.hp.d_tin == self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Vmask, Tmask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)
        elif self.hp.d_vin == self.d_modal1 and self.hp.d_ain == self.d_modal2:
            encoded = self.net(modal1, modal2, modal2, Vmask, Amask, whether_add_position=whether_add_position,counterfactual_attention_type=counterfactual_attention_type)

        elif self.d_modal1 == self.d_modal2 ==self.hp.d_tin:
            encoded = self.net(modal1, modal2, modal2, Tmask, None, whether_add_position=whether_add_position)
        elif self.d_modal1 == self.d_modal2 ==self.hp.d_ain:
            encoded = self.net(modal1, modal2, modal2, Amask, None, whether_add_position=whether_add_position)
        elif self.d_modal1 == self.d_modal2 ==self.hp.d_vin:
            encoded = self.net(modal1, modal2, modal2, Vmask, None, whether_add_position=whether_add_position)

        output = self.output_linear(F.relu(self.linear(encoded)))
        return output


# √
class SelfAttention(nn.Module):
    def __init__(self, hp, d_in, d_model, nhead, dim_feedforward, dropout, num_layers=6):
        super(SelfAttention, self).__init__()
        self.hp = hp
        self.d_in = d_in
        self.num_heads = nhead
        self.d_model = d_model
        self.proj = nn.Conv1d(self.d_in, self.d_model, kernel_size=1, padding=0, bias=False)
        self.layers = num_layers
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.output_linear = nn.Linear(dim_feedforward, d_in)
        self.attn_dropout = dropout
        self.relu_dropout = self.hp.relu_dropout
        self.res_dropout = self.hp.res_dropout
        self.embed_dropout = self.hp.embed_dropout
        self.attn_mask = self.hp.attn_mask

        self.net = self.get_network()


    def get_network(self, layers=-1):
        return TransformerEncoder(embed_dim=self.d_model, 
                           num_heads=self.num_heads,
                           layers=max(self.layers, layers),
                           attn_dropout=self.attn_dropout,
                           relu_dropout=self.relu_dropout,
                           res_dropout=self.res_dropout,
                           embed_dropout=self.embed_dropout,
                           attn_mask=self.attn_mask)
    def forward(self, input, maskT=None, maskA=None, maskV=None, whether_add_position=True):
        """
        传入格式：(seq_len, batch_size,emb_size)
        """
        input = self.proj(input.permute(1, 2, 0))
        x = input.permute(2, 0, 1)
        if self.d_in == self.hp.d_ain:
            encoded = self.net(x,maskA,whether_add_position=whether_add_position)
        elif self.d_in == self.hp.d_vin:
            encoded = self.net(x, maskV,whether_add_position=whether_add_position)
        elif self.d_in == self.hp.d_tin:
            encoded = self.net(x,maskT,whether_add_position=whether_add_position)
        output = self.output_linear(self.linear(encoded))
        return output


# √
class FinalFusionSelfAttention(nn.Module):
    def __init__(self,hp, text_in_dim, audio_in_dim, vision_in_dim, 
                cross_ta_dim, cross_tv_dim, 
                d_model, nhead, dim_feedforward, 
                dropout, num_layers=6, n_class=1):
        super(FinalFusionSelfAttention, self).__init__()
        self.hp=hp
        self.text_in_dim=text_in_dim
        self.audio_in_dim=audio_in_dim
        self.vision_in_dim=vision_in_dim
        self.cross_ta_dim=cross_ta_dim
        self.cross_tv_dim=cross_tv_dim

        self.num_heads = nhead

        self.d_model=d_model

        self.proj_t_to_model = nn.Conv1d(self.text_in_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_a_to_model = nn.Conv1d(self.audio_in_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_v_to_model = nn.Conv1d(self.vision_in_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_ta_to_model = nn.Conv1d(self.cross_tv_dim, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_tv_to_model = nn.Conv1d(self.cross_ta_dim, self.d_model, kernel_size=1, padding=0, bias=False)

        self.layers = num_layers

        self.attn_dropout=dropout
        self.relu_dropout=self.hp.relu_dropout
        self.res_dropout=self.hp.res_dropout
        self.embed_dropout=self.hp.embed_dropout
        self.attn_mask=self.hp.attn_mask
        self.n_class=n_class

        self.linear = nn.Linear(d_model, dim_feedforward)
        self.proj_linear=nn.Linear(dim_feedforward, hp.d_prjh)
        self.output_linear=nn.Linear(hp.d_prjh, self.n_class)

        self.net=self.get_network()

    def get_network(self, layers=-1):
        return TransformerEncoder(embed_dim=self.d_model, 
                           num_heads=self.num_heads,
                           layers=max(self.layers, layers),
                           attn_dropout=self.attn_dropout,
                           relu_dropout=self.relu_dropout,
                           res_dropout=self.res_dropout,
                           embed_dropout=self.embed_dropout,
                           attn_mask=self.attn_mask)

    def forward(self,text,audio,vision,ta,tv):
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        """
        text = self.proj_t_to_model(text.permute(1, 2, 0))
        text=text.permute(2, 0, 1)##文本

        audio= self.proj_a_to_model(audio.permute(1, 2, 0))
        audio=audio.permute(2, 0, 1)##音频

        vision = self.proj_v_to_model(vision.permute(1, 2, 0))
        vision=vision.permute(2, 0, 1)##视频

        ta = self.proj_ta_to_model(ta.permute(1, 2, 0))
        ta=ta.permute(2, 0, 1)##跨模态ta

        tv = self.proj_tv_to_model(tv.permute(1, 2, 0))
        tv=tv.permute(2, 0, 1)##跨模态tv

        final_input=torch.cat([text,audio,vision,ta,tv],dim=0)

        encoded=self.net(final_input,final_input,final_input)
        pooled_encoded=torch.mean(encoded,dim=0)#[bs,dim]
        fusion=self.proj_linear(F.relu(self.linear(pooled_encoded)))
        pred=self.output_linear(fusion)

        return fusion, pred


# √
class CLUB(nn.Module):
    """
        Compute the Contrastive Log-ratio Upper Bound (CLUB) given a pair of inputs.
        Refer to https://arxiv.org/pdf/2006.12013.pdf and https://github.com/Linear95/CLUB/blob/f3457fc250a5773a6c476d79cda8cb07e1621313/MI_DA/MNISTModel_DANN.py#L233-254

        Args:
            hidden_size(int): embedding size
            activation(int): the activation function in the middle layer of MLP
    """

    def __init__(self, hidden_size, activation='Tanh'):
        super(CLUB, self).__init__()
        try:
            self.activation = getattr(nn, activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
            
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.eps = 1e-6

    # def forward(self, modal_a, modal_b, sample=False):
    #     """
    #         CLUB with random shuffle, the Q function in original paper:
    #             CLUB = E_p(x,y)[log q(y|x)]-E_p(x)p(y)[log q(y|x)]
            
    #         Args:
    #             modal_a (Tensor): x in above equation
    #             model_b (Tensor): y in above equation
    #     """
    #     mu, logvar = self.mlp_mu(modal_a), self.mlp_logvar(modal_a)
    #     batch_size = mu.size(0)
    #     pred = mu

    #     # pred b using a
    #     pred_tile = mu.unsqueeze(1).repeat(1, batch_size, 1)  # (bs, bs, emb_size)
    #     true_tile = pred.unsqueeze(0).repeat(batch_size, 1, 1)  # (bs, bs, emb_size)

    #     positive = - (mu - modal_b) ** 2 / 2. / torch.exp(logvar)
    #     negative = - torch.mean((true_tile - pred_tile) ** 2, dim=1) / 2. / torch.exp(logvar)

    #     lld = torch.mean(torch.sum(positive, -1))
    #     bound = torch.mean(torch.sum(positive, -1) - torch.sum(negative, -1))
    #     return lld, bound
    def forward(self, modal_a, modal_b, sample=False):
        mu, logvar = self.mlp_mu(modal_a), self.mlp_logvar(modal_a)  # (B, D)
        var = torch.exp(logvar) + self.eps

        # 正项：匹配对 (x_i, y_i)
        positive = - (modal_b - mu).pow(2) / (2.0 * var)             # (B, D)

        # 负项（任选其一）：

        # # 1) 全配对 + 广播
        # diff = modal_b.unsqueeze(0) - mu.unsqueeze(1)                # (B, B, D)
        # negative = - diff.pow(2).mean(dim=1) / (2.0 * var)           # (B, D)

        # 2) 或者：随机打乱
        perm = torch.randperm(mu.size(0), device=mu.device)
        negative = - (modal_b[perm] - mu).pow(2) / (2.0 * var)     # (B, D)

        lld = positive.sum(-1).mean()
        bound = (positive.sum(-1) - negative.sum(-1)).mean()
        return lld, bound


# √
class MMILB(nn.Module):  # 双模态表示的互信息下界
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """

    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x)  # (bs, hidden_size)

        positive = -(mu - y) ** 2 / 2. / torch.exp(logvar)  
        lld = torch.mean(torch.sum(positive, -1))  

        # For Gaussian Distribution Estimation 高斯分布估计
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos': None, 'neg': None}

        if labels is not None:
            y = self.entropy_prj(y)
            pos_y = y[labels.squeeze() > 0]
            neg_y = y[labels.squeeze() < 0]

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']

                # Diagonal setting            
                # pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                # neg_all = torch.cat(neg_history + [neg_y], dim=0)
                # mu_pos = pos_all.mean(dim=0)
                # mu_neg = neg_all.mean(dim=0)

                # sigma_pos = torch.mean(pos_all ** 2, dim = 0) - mu_pos ** 2 # (embed)
                # sigma_neg = torch.mean(neg_all ** 2, dim = 0) - mu_neg ** 2 # (embed)
                # H = 0.25 * (torch.sum(torch.log(sigma_pos)) + torch.sum(torch.log(sigma_neg)))

                # compute the entire co-variance matrix
                pos_all = torch.cat(pos_history + [pos_y], dim=0)  # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)
                sigma_pos = torch.mean(torch.bmm((pos_all - mu_pos).unsqueeze(-1), (pos_all - mu_pos).unsqueeze(1)), dim=0)
                sigma_neg = torch.mean(torch.bmm((neg_all - mu_neg).unsqueeze(-1), (neg_all - mu_neg).unsqueeze(1)), dim=0)
                H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))  

        return lld, sample_dict, H


# √
class CPC(nn.Module):  # 对比预测编码
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """

    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        # x是：t a v     y是融合后的
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.activation = getattr(nn, activation)
        if n_layers == 1: 
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)

    def forward(self, x, y):
        """Calulate the score
            eg： 
            nce_t = self.cpc_zt(text, fusion)  # 3.4660
            x: torch.Size([32, 768])
            y: torch.Size([32, 128])
        """
        x_pred = self.net(y)  

        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)  
        x = x / x.norm(dim=1, keepdim=True) 

        pos = torch.sum(x * x_pred, dim=-1)
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1) 
        nce = -(pos - neg).mean()
        return nce


# √
class RNNEncoder(nn.Module):  # 视频和音频的特征提取
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1) * hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        eg: self.visual_enc(visual, v_len) # torch.Size([134, 32, 5])
        '''
        lengths = lengths.to(torch.int64)  

        # 将序列送给 RNN 进行处理之前，需要采用 pack_padded_sequence 进行压缩，压缩掉无效的填充值
        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        
        _, final_states = self.rnn(packed_sequence)

        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else: 
            h = self.dropout(final_states[0].squeeze())

        y_1 = self.linear_1(h)
        return y_1


# √
class Encoder(nn.Module):
    def __init__(self, d_in, d_model, nhead, dim_feedforward, dropout, activation="relu", num_layers=6):
        super().__init__()
        self.d_in=d_in
        self.num_heads = nhead
        self.pe = PositionalEncoding(d_model, dropout)
        self.proj = nn.Conv1d(self.d_in, d_model, kernel_size=1, padding=0, bias=False)
        self.layer = TransEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransEncoder(self.layer, num_layers)
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.output_linear = nn.Linear(dim_feedforward, d_in)

    def forward(self, inputs, attn_mask):
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        """
        inputs_proj = self.proj(inputs.permute(1, 2, 0))
        inputs_proj = inputs_proj.permute(2,0,1)
        inputs = self.pe(inputs_proj)
        #attn_mask=buffered_future_mask(inputs) if attn_mask else None
        attn_mask = attn_mask.transpose(1, 0)
        attn_mask = compute_mask(attn_mask, attn_mask, self.num_heads)
        encoded = self.encoder(inputs, mask=attn_mask)
        encoded = self.linear(encoded)
        encoded = self.output_linear(F.relu(encoded))
        return encoded


# √
def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prior, after):
        return self.norm(prior + self.dropout(after))


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, inputs):
        return self.linear2(self.dropout(self.activation(self.linear1(inputs))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def drop_path(paths, drop_rates):
    lens = len(paths)
    drop_rates = torch.tensor(drop_rates)
    drop = torch.bernoulli(drop_rates)
    if torch.all(drop == 0):
        idx = randint(0, lens-1)
        output = paths[idx]
    else:
        output = sum([paths[i] * drop[i] for i in range(lens)]) / torch.sum(drop)
    return output


def compute_mask(mask_1, mask_2, num_heads):
    mask_1 = torch.unsqueeze(mask_1, 2)
    mask_2 = torch.unsqueeze(mask_2, 1)
    attn_mask = torch.bmm(mask_1, mask_2)
    attn_mask = attn_mask.repeat(num_heads, 1, 1)
    return attn_mask.bool()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# -------------------------------
# 变分头：任意输入维 → 共享潜在维 z_dim
# 对角高斯近似：输出 (mu, logvar)，形状 [L,B,z_dim]
# -------------------------------
class VarHeadZ(nn.Module):
    """
    将输入逐 token 投影到共享潜在空间 z_dim，并产生对角高斯参数（mu, logσ²）

    输入:
        x: [L, B, D_in] 或 [N, D_in]
    输出:
        mu:     [..., z_dim]
        logvar: [..., z_dim]
    """
    def __init__(self, D_in: int, z_dim: int = 32, hidden: Optional[int] = None,
                 dropout: float = 0.1, ln_eps: float = 1e-5):
        super().__init__()
        self.z_dim = int(z_dim)
        hidden = D_in if hidden is None else hidden

        self.ff = nn.Sequential(
            nn.Linear(D_in, hidden),
            nn.GELU(),                # 可改为 SiLU/ReLU；GELU 在小头里更平滑
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden, eps=ln_eps),
        )
        self.mu_head     = nn.Linear(hidden, self.z_dim)
        self.logvar_head = nn.Linear(hidden, self.z_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # 初始 σ≈exp(-1) ~ 0.37

    def _forward_nd(self, x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.ff(x2d)                  # [..., hidden]
        mu = self.mu_head(y)              # [..., z_dim]
        logvar = self.logvar_head(y)      # [..., z_dim]
        return mu, logvar

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:                  # [L,B,D_in]
            L, B, D = x.shape
            x2d = x.reshape(L * B, D)
            mu, logvar = self._forward_nd(x2d)
            return mu.view(L, B, self.z_dim), logvar.view(L, B, self.z_dim)
        elif x.dim() == 2:                # [N,D_in]
            return self._forward_nd(x)
        else:
            raise ValueError("VarHeadZ expects [L,B,D_in] or [N,D_in]")


# -------------------------------
# APF 模块（Linear + Orthogonal + LayerNorm 版）
# -------------------------------
class APF(nn.Module):
    """
    Ambiguity-Perception Fusion

    放置位置：跨模态注意力之后、任何池化之前。

    forward 输入:
        Ft   : [L,B,D_tin]        文本特征（例如 768）
        Fvt  : [L,B,d_model]      视觉→文本（CrossAttention 输出）
        Fat  : [L,B,d_model]      语音→文本（CrossAttention 输出）
        maskT: [L,B]              文本时间轴的 pad 掩码（1=pad 无效，0=有效）
        train_step: bool          仅训练步时更新 EMA 分母
        progress 或 (step,total_steps): τ 的线性调度（二选一，任选其一提供）

    forward 输出:
        F_apf : [L,B,d_model]     融合后的序列特征
        monitor: Dict             监控指标（用于日志/告警）
    """

    def __init__(self,
                 D_tin: int,          # 文本原始维度（例如 768）
                 d_model: int,        # 任务空间维度（与 CrossAttention 输出一致，≈30~40，最多不超过 256）
                 z_dim: int = 32,     # 共享潜空间维度（KL 在此维度上计算）
                 clamp_min: float = -8.0,
                 clamp_max: float = 2.0,
                 tau_start: float = 0.7,
                 tau_end: float = 0.5,             # 训练前 30% 步线性降到 0.5（建议）
                 beta: float = 0.99,               # EMA 动量（归一化分母）
                 ln_eps: float = 1e-5,
                 use_LN_before_varhead: bool = True,   # VarHeadZ 前是否接 LN（稳定建议开启）
                 use_proj_LN_in: bool = True,          # 文本投影前的 LN
                 use_proj_LN_out: bool = True,         # 文本投影后的 LN
                 eps: float = 1e-6                      # 数值稳定常数
                 ):
        super().__init__()
        self.D_tin = int(D_tin)
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.beta = float(beta)
        self.eps = float(eps)

        # ---- 文本 768 → d_model 的 Linear 投影（正交初始化）----
        self.proj_ln_in  = nn.LayerNorm(self.D_tin,  eps=ln_eps) if use_proj_LN_in  else nn.Identity()
        self.text_proj   = nn.Linear(self.D_tin, self.d_model, bias=False)
        self.proj_ln_out = nn.LayerNorm(self.d_model, eps=ln_eps) if use_proj_LN_out else nn.Identity()
        self._init_linear_orthogonal(self.text_proj)

        # ---- 变分头：在各自原空间产生 (μ, logσ²)；三路参数不共享，输出维 z_dim 一致 ----
        self.pre_ln_t  = nn.LayerNorm(self.D_tin,  eps=ln_eps) if use_LN_before_varhead else nn.Identity()
        self.pre_ln_vt = nn.LayerNorm(self.d_model, eps=ln_eps) if use_LN_before_varhead else nn.Identity()
        self.pre_ln_at = nn.LayerNorm(self.d_model, eps=ln_eps) if use_LN_before_varhead else nn.Identity()

        self.z_t  = VarHeadZ(D_in=self.D_tin,   z_dim=self.z_dim, ln_eps=ln_eps)
        self.z_vt = VarHeadZ(D_in=self.d_model, z_dim=self.z_dim, ln_eps=ln_eps)
        self.z_at = VarHeadZ(D_in=self.d_model, z_dim=self.z_dim, ln_eps=ln_eps)

        # ---- EMA 分母（视觉/语音各一条），register_buffer 使其随模型保存且不被优化器更新 ----
        self.register_buffer("emaKv", torch.tensor(1.0))
        self.register_buffer("emaKa", torch.tensor(1.0))
        self.register_buffer("last_tau", torch.tensor(self.tau_start))  # 记录上次 τ（仅监控用途）

    # ---------- 初始化 & 工具 ----------
    @staticmethod
    def _init_linear_orthogonal(linear: nn.Linear, gain: float = 1.0):
        """正交初始化线性层权重（行/列正交，稳定强降维；无 bias）。"""
        nn.init.orthogonal_(linear.weight, gain=gain)

    @staticmethod
    def _to_valid_mask_from_T(maskT: torch.Tensor, like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 maskT(1=pad,0=valid) 转成：
            valid_bool: [L,B] True=有效
            valid_f:    [L,B] float(1=有效,0=pad)
        """
        if maskT.dtype == torch.bool:
            valid_bool = ~maskT
            valid_f = (~maskT).to(dtype=like.dtype)
        else:
            valid_f = (maskT == 0).to(dtype=like.dtype)
            valid_bool = (valid_f > 0.5)
        return valid_bool, valid_f

    @staticmethod
    def _kl_diag_gauss(mu1: torch.Tensor, var1: torch.Tensor,
                       mu2: torch.Tensor, var2: torch.Tensor) -> torch.Tensor:
        """
        KL(N1||N2)，对角高斯，沿最后一维求和 → [L,B]
        0.5 * sum( log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 - 1 )
        """
        return 0.5 * (
            (var1 / var2)
            + ((mu1 - mu2) ** 2) / var2
            + (var2.log() - var1.log())
            - 1.0
        ).sum(dim=-1)

    @staticmethod
    def _safe_mean(x: torch.Tensor, valid_bool: torch.Tensor) -> float:
        n = int(valid_bool.sum().item())
        return (x[valid_bool].float().mean().item() if n > 0 else 0.0)

    def _current_tau(self,
                     progress: Optional[float] = None,
                     step: Optional[int] = None,
                     total_steps: Optional[int] = None) -> float:
        """
        线性调度 τ：tau(progress) = tau_start * (1 - p) + tau_end * p
        若未提供 progress 或 (step,total_steps)，则返回 tau_start。
        """
        if progress is None and (step is not None and total_steps is not None and total_steps > 0):
            progress = max(0.0, min(1.0, float(step) / float(total_steps)))
        if progress is None:
            tau = self.tau_start
        else:
            p = max(0.0, min(1.0, float(progress)))
            tau = self.tau_start * (1.0 - p) + self.tau_end * p
        # 记录（便于监控）
        self.last_tau = torch.as_tensor(tau, device=self.last_tau.device, dtype=self.last_tau.dtype)
        return tau

    # ---------- 前向 ----------
    def forward(self,
                Ft: torch.Tensor,          # [L,B,D_tin]
                Fvt: torch.Tensor,         # [L,B,d_model]
                Fat: torch.Tensor,         # [L,B,d_model]
                maskT: torch.Tensor,       # [L,B] 1=pad
                train_step: bool = False,
                progress: Optional[float] = None,
                step: Optional[int] = None,
                total_steps: Optional[int] = None
                ) -> Tuple[torch.Tensor, Dict]:
        L, B, Dt = Ft.shape
        assert Dt == self.D_tin, f"Ft last dim {Dt} != D_tin {self.D_tin}"
        assert Fvt.shape[:2] == (L, B) and Fat.shape[:2] == (L, B), "Fvt/Fat must align on [L,B]"
        assert Fvt.shape[2] == self.d_model and Fat.shape[2] == self.d_model, "Fvt/Fat last dim must equal d_model"

        # ------ 1) 潜空间：三路各自产出 (mu, logvar) 并计算对称 KL ------
        Ft_ln  = self.pre_ln_t(Ft)      # [L,B,D_tin]
        Fvt_ln = self.pre_ln_vt(Fvt)    # [L,B,d_model]
        Fat_ln = self.pre_ln_at(Fat)    # [L,B,d_model]

        mu_t,  logv_t  = self.z_t(Ft_ln)     # [L,B,z_dim]
        mu_vt, logv_vt = self.z_vt(Fvt_ln)   # [L,B,z_dim]
        mu_at, logv_at = self.z_at(Fat_ln)   # [L,B,z_dim]

        # clamp + 方差
        logv_t  = logv_t.clamp(self.clamp_min, self.clamp_max)
        logv_vt = logv_vt.clamp(self.clamp_min, self.clamp_max)
        logv_at = logv_at.clamp(self.clamp_min, self.clamp_max)

        var_t, var_vt, var_at = logv_t.exp() + self.eps, logv_vt.exp() + self.eps, logv_at.exp() + self.eps

        # 对称 KL（逐 token）
        Kv = 0.5 * ( self._kl_diag_gauss(mu_t, var_t, mu_vt, var_vt) +
                     self._kl_diag_gauss(mu_vt, var_vt, mu_t, var_t) )  # [L,B]
        Ka = 0.5 * ( self._kl_diag_gauss(mu_t, var_t, mu_at, var_at) +
                     self._kl_diag_gauss(mu_at, var_at, mu_t, var_t) )

        # 有效位掩码（maskT: 1=pad）
        valid_bool, valid_f = self._to_valid_mask_from_T(maskT, Kv)

        # EMA 分母更新（仅训练步）
        if self.training and train_step:
            denom = valid_f.sum().clamp(min=1.0)
            bmKv = (Kv * valid_f).sum() / denom
            bmKa = (Ka * valid_f).sum() / denom
            self.emaKv = self.beta * self.emaKv + (1.0 - self.beta) * bmKv.detach()
            self.emaKa = self.beta * self.emaKa + (1.0 - self.beta) * bmKa.detach()

        # τ（线性调度）
        tau_now = self._current_tau(progress=progress, step=step, total_steps=total_steps)

        # 门控权重（歧义越大越“退回文本”）
        wv = torch.sigmoid((Kv / (self.emaKv + self.eps)) / tau_now) * valid_f  # [L,B]
        wa = torch.sigmoid((Ka / (self.emaKa + self.eps)) / tau_now) * valid_f  # [L,B]

        # ------ 2) 任务空间融合：文本投影到 d_model（Linear + Orthogonal + LN），再按 w 融合 ------
        text_proj = self.proj_ln_out(self.text_proj(self.proj_ln_in(Ft)))  # [L,B,d_model]

        # 逐 token 融合
        F_apf = (1.0 - wv.unsqueeze(-1)) * Fvt \
              + (1.0 - wa.unsqueeze(-1)) * Fat \
              + (wv + wa).unsqueeze(-1)   * text_proj    # [L,B,d_model]

        # ------ 3) 监控指标（仅统计有效位） ------
        # 监控项（返回给上层记录）——严格全局统计所需的“加和+计数”
        valid_bool, _ = self._to_valid_mask_from_T(maskT, Kv)
        n_valid = int(valid_bool.sum().item())
        sum_wv = float(wv[valid_bool].sum().item())
        sum_wa = float(wa[valid_bool].sum().item())
        sum_Kv = float(Kv[valid_bool].sum().item())
        sum_Ka = float(Ka[valid_bool].sum().item())

        monitor = {
            # 加权汇总所需
            "n_valid": n_valid,
            "sum_wv": sum_wv, "sum_wa": sum_wa,
            "sum_Kv": sum_Kv, "sum_Ka": sum_Ka,

            # 便于快速看区间均值/极端比例（保持原有）
            "w_v_mean": (sum_wv / max(n_valid, 1)),
            "w_a_mean": (sum_wa / max(n_valid, 1)),
            "Kv_mean":  (sum_Kv / max(n_valid, 1)),
            "Ka_mean":  (sum_Ka / max(n_valid, 1)),
            "frac_wv_0": float(((wv[valid_bool] < 0.05).float().mean().item()) if n_valid>0 else 0.0),
            "frac_wv_1": float(((wv[valid_bool] > 0.95).float().mean().item()) if n_valid>0 else 0.0),
            "frac_wa_0": float(((wa[valid_bool] < 0.05).float().mean().item()) if n_valid>0 else 0.0),
            "frac_wa_1": float(((wa[valid_bool] > 0.95).float().mean().item()) if n_valid>0 else 0.0),

            # 参照尺度
            "tau_now": float(tau_now),
            "emaKv":   float(self.emaKv.item()),
            "emaKa":   float(self.emaKa.item()),
        }

        return F_apf, monitor