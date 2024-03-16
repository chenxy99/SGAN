import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.mlp import MLP, TwoLayerMLP
from lib.modules.aggr.attention import ImageAttention, CaptionAttention
from transformers import BertModel
from typing import Optional, Tuple, Union
import numpy as np
import logging

from lib.modules.graph.gat_conv import GATv2ConvCustomv1, GATv2ConvCustomv2
from torch_geometric.utils import to_dense_adj, dense_to_sparse


logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def get_text_encoder(embed_size):
    return EncoderText(embed_size)

def get_image_encoder(data_name, img_dim, embed_size, opt, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

def get_graph_encoder(embed_size, graph_layer_num=5, graph_head_num=1, opt=None):
    return EncoderGraph(embed_size=embed_size, graph_layer_num=graph_layer_num, graph_head_num=graph_head_num, opt=opt)

# Vision Model with ResNet
class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic'):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)
        features = self.image_encoder(base_features)
        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))

class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic'):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(images) + features

        return features

# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)

        self.freeze_unused_parameters()

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for modules in [self.linear.modules()]:
            for m in modules:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features +
                                              m.out_features)
                    m.weight.data.uniform_(-r, r)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def freeze_unused_parameters(self):
        # pooler will not be used when train the bert
        for p in self.bert.pooler.parameters():
            p.require_grad = False

    def forward(self, x):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D

        cap_emb = self.linear(bert_emb)

        return cap_emb

# Graph Model
class EncoderGraph(nn.Module):
    def __init__(self, embed_size, graph_layer_num, graph_head_num, opt):
        super(EncoderGraph, self).__init__()
        self.opt = opt
        self.embed_size = embed_size
        self.graph_layer_num = graph_layer_num
        self.graph_head_num = graph_head_num
        self.end_node_feat = nn.Linear(self.embed_size, self.embed_size)

        self.Wg = nn.Linear(self.embed_size, self.embed_size)

        self.img_att = ImageAttention(self.embed_size, self.embed_size, self.embed_size // 4)
        self.cap_att = CaptionAttention(self.embed_size, self.embed_size, self.embed_size // 4)

        in_channels, hidden_channels, heads = self.embed_size, self.embed_size // self.graph_head_num, self.graph_head_num

        if not self.opt.no_norm:
            # image graph encoder
            self.img_conv_list = [GATv2ConvCustomv2(in_channels, hidden_channels, heads=heads, add_self_loops=False, no_norm=self.opt.no_norm, gamma=self.opt.gamma)]
            for idx in range(self.graph_layer_num - 1):
                self.img_conv_list.append(GATv2ConvCustomv2(hidden_channels * heads, hidden_channels, heads=heads, add_self_loops=False, no_norm=self.opt.no_norm, gamma=self.opt.gamma))
            self.img_conv_list = nn.ModuleList(self.img_conv_list)

            # caption graph encoder
            self.cap_conv_list = [GATv2ConvCustomv2(in_channels, hidden_channels, heads=heads, add_self_loops=False, no_norm=self.opt.no_norm, gamma=self.opt.gamma)]
            for idx in range(self.graph_layer_num - 1):
                self.cap_conv_list.append(GATv2ConvCustomv2(hidden_channels * heads, hidden_channels, heads=heads, add_self_loops=False, no_norm=self.opt.no_norm, gamma=self.opt.gamma))
            self.cap_conv_list = nn.ModuleList(self.cap_conv_list)
        else:
            # image graph encoder
            self.img_conv_list = [GATv2ConvCustomv1(in_channels, hidden_channels, heads=heads, add_self_loops=False)]
            for idx in range(self.graph_layer_num - 1):
                self.img_conv_list.append(GATv2ConvCustomv1(hidden_channels * heads, hidden_channels, heads=heads, add_self_loops=False))
            self.img_conv_list = nn.ModuleList(self.img_conv_list)

            # caption graph encoder
            self.cap_conv_list = [GATv2ConvCustomv1(in_channels, hidden_channels, heads=heads, add_self_loops=False)]
            for idx in range(self.graph_layer_num - 1):
                self.cap_conv_list.append(GATv2ConvCustomv1(hidden_channels * heads, hidden_channels, heads=heads, add_self_loops=False))
            self.cap_conv_list = nn.ModuleList(self.cap_conv_list)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for modules in [self.Wg.modules(), self.img_att.modules(), self.cap_att.modules()]:
            for m in modules:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features +
                                              m.out_features)
                    m.weight.data.uniform_(-r, r)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, goal_feature, cap_feature, img_feature, cap_emb_mask, edge_index):
        # image propagation
        image_attention_weights_list = []
        image_adjacency_matrix_list = []
        for idx in range(self.graph_layer_num):
            if idx == 0:
                current_goal_feature = self.Wg(goal_feature)
                current_end_node_feature = self.end_node_feat(goal_feature)
            else:
                current_goal_feature = node_feature[0].unsqueeze(0)
                current_end_node_feature = node_feature[1].unsqueeze(0)
            img_attention_weights = self.img_att(img_feature, current_goal_feature)
            img_aggr_features = (img_feature * img_attention_weights.unsqueeze(-1)).sum(1)
            prev_node_feature = torch.cat([current_goal_feature, current_end_node_feature, img_aggr_features])
            if not self.opt.no_norm:
                prev_node_feature = l2norm(prev_node_feature, -1)
            node_feature, attention_weights_info = self.img_conv_list[idx](x=prev_node_feature, edge_index=edge_index, return_attention_weights=True)
            node_feature += prev_node_feature
            node_feature = F.elu(node_feature)
            node_feature = F.dropout(node_feature, training=self.training)

            # recode attention for image
            image_attention_weights_list.append(img_attention_weights)
            # recode den_adj for node relationship
            dense_adj = to_dense_adj(edge_index=attention_weights_info[0], batch=None,
                                     edge_attr=attention_weights_info[1], max_num_nodes=node_feature.shape[0])
            dense_adj = dense_adj.squeeze()
            # dense_adj = dense_adj / (dense_adj.max(-1, keepdim=True)[0] + eps)
            # if self.opt.two_stage == True:
            #     task_retrieve_prob = dense_adj[0, 2:].data
            #     dense_adj[1:, 2:] = dense_adj[1:, 2:] * task_retrieve_prob.unsqueeze(0)
            #     dense_adj[2:, 1:] = dense_adj[2:, 1:] * task_retrieve_prob.unsqueeze(1)
            image_adjacency_matrix_list.append(dense_adj)

        # image propagation
        caption_attention_weights_list = []
        caption_adjacency_matrix_list = []
        for idx in range(self.graph_layer_num):
            if idx == 0:
                current_goal_feature = self.Wg(goal_feature)
                current_end_node_feature = self.end_node_feat(goal_feature)
            else:
                current_goal_feature = node_feature[0].unsqueeze(0)
                current_end_node_feature = node_feature[1].unsqueeze(0)
            cap_attention_weights = self.cap_att(cap_feature, current_goal_feature, cap_emb_mask)
            cap_aggr_features = (cap_feature * cap_attention_weights.unsqueeze(-1)).sum(1)
            prev_node_feature = torch.cat([current_goal_feature, current_end_node_feature, cap_aggr_features])
            if not self.opt.no_norm:
                prev_node_feature = l2norm(prev_node_feature, -1)
            node_feature, attention_weights_info = self.img_conv_list[idx](x=prev_node_feature, edge_index=edge_index,
                                                                           return_attention_weights=True)
            node_feature += prev_node_feature
            node_feature = F.elu(node_feature)
            node_feature = F.dropout(node_feature, training=self.training)

            # recode attention for caption
            caption_attention_weights_list.append(cap_attention_weights)
            # recode den_adj for node relationship
            dense_adj = to_dense_adj(edge_index=attention_weights_info[0], batch=None,
                                     edge_attr=attention_weights_info[1], max_num_nodes=node_feature.shape[0])
            dense_adj = dense_adj.squeeze()
            # dense_adj = dense_adj / (dense_adj.max(-1, keepdim=True)[0] + eps)
            # if self.opt.two_stage == True:
            #     task_retrieve_prob = dense_adj[0, 2:].data
            #     dense_adj[1:, 2:] = dense_adj[1:, 2:] * task_retrieve_prob.unsqueeze(0)
            #     dense_adj[2:, 1:] = dense_adj[2:, 1:] * task_retrieve_prob.unsqueeze(1)
            caption_adjacency_matrix_list.append(dense_adj)

        return  image_attention_weights_list, image_adjacency_matrix_list, \
                caption_attention_weights_list, caption_adjacency_matrix_list
