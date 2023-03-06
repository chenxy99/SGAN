import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import scipy.stats

from lib.encoders import get_image_encoder, get_text_encoder, get_graph_encoder
from lib.loss import BaselineLoss, FocalLoss, OHEMLoss, AttentionLoss, BaselineTwoStageLoss, RankingLoss, \
    OHEMTwoStageLoss
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

epsilon = 1e-7

import logging
logger = logging.getLogger(__name__)


class BaselineModel(nn.Module):
    # initializers
    def __init__(self, opt):
        super(BaselineModel, self).__init__()
        self.opt = opt

        self.goal_enc = get_text_encoder(embed_size=opt.embed_size)
        self.img_enc = get_image_encoder(data_name=opt.data_name, img_dim=opt.img_dim,
                                         embed_size=opt.embed_size, opt=opt,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path)
        self.cap_enc = get_text_encoder(embed_size=opt.embed_size)
        self.graph_enc = get_graph_encoder(embed_size=opt.embed_size,
                                                  graph_layer_num=opt.graph_layer_num,
                                                  graph_head_num=opt.graph_head_num,
                                                  opt=opt)
        if opt.loss_type == "bce":
            self.baseline_loss = BaselineLoss(opt=opt)
            if opt.two_stage == True:
                self.baseline_loss = BaselineTwoStageLoss(opt=opt)
        elif opt.loss_type == "focalloss":
            self.baseline_loss = FocalLoss(opt=opt)
        elif opt.loss_type == "ohem":
            self.baseline_loss = OHEMLoss(opt=opt, ratio=opt.ohem_ratio)
            if opt.two_stage == True:
                self.baseline_loss = OHEMTwoStageLoss(opt=opt)
        elif opt.loss_type == "ranking":
            self.baseline_loss = RankingLoss(opt=opt)
        else:
            raise ValueError('Invalid loss_type argument {}'.format(opt.loss_type))

        self.attention_loss = AttentionLoss(opt=opt)


    def forward(self, goal, caption_steps, images, actual_problem_solving_step_indicator,
                aggregate_caption_attentions=None, scale_down_aggregate_attention_maps=None, topological_graph_matrix=None):
        if aggregate_caption_attentions is not None:
            aggregate_caption_attentions = aggregate_caption_attentions.view(aggregate_caption_attentions.shape[0],
                                                                             aggregate_caption_attentions.shape[1], -1)
        if scale_down_aggregate_attention_maps is not None:
            scale_down_aggregate_attention_maps = scale_down_aggregate_attention_maps.\
                view(scale_down_aggregate_attention_maps.shape[0], scale_down_aggregate_attention_maps.shape[1], -1)
        if self.training:
            predicts = self.training_process(goal, caption_steps, images, actual_problem_solving_step_indicator,
                aggregate_caption_attentions, scale_down_aggregate_attention_maps, topological_graph_matrix)
        else:
            predicts = self.inference(goal, caption_steps, images, actual_problem_solving_step_indicator,
                aggregate_caption_attentions, scale_down_aggregate_attention_maps, topological_graph_matrix)

        # for this case the predicts result is padding at the end of each candidate solution steps
        return predicts


    def training_process(self, goal, caption_steps, images, actual_problem_solving_step_indicator,
                aggregate_caption_attentions=None, scale_down_aggregate_attention_maps=None, topological_graph_matrix=None):
        batch_size = goal.shape[0]
        batch_step_num = actual_problem_solving_step_indicator.sum(-1)
        actual_total_step_num = batch_step_num.sum()
        # batch_step_total_num = actual_total_step_num + padding num
        batch_step_total_num = actual_problem_solving_step_indicator.shape[0] * actual_problem_solving_step_indicator.shape[1]
        # construct the topological_graph_matrix for the current batch
        if topological_graph_matrix is not None:
            batch_topological_graph_matrix = topological_graph_matrix.new_zeros(batch_size, batch_step_total_num, batch_step_total_num)
            start_idx = 2
            for idx in range(batch_size):
                batch_topological_graph_matrix[idx, :2, :2] = topological_graph_matrix[idx, :2, :2]
                batch_topological_graph_matrix[idx, start_idx:start_idx + batch_step_num[idx], :2] = \
                    topological_graph_matrix[idx, 2:2 + batch_step_num[idx], :2]
                batch_topological_graph_matrix[idx, :2, start_idx:start_idx + batch_step_num[idx]] = \
                    topological_graph_matrix[idx, :2, 2:2 + batch_step_num[idx]]
                batch_topological_graph_matrix[idx, start_idx:start_idx + batch_step_num[idx], start_idx:start_idx + batch_step_num[idx]] = \
                    topological_graph_matrix[idx, 2:2 + batch_step_num[idx], 2:2 + batch_step_num[idx]]
                start_idx += batch_step_num[idx]
        else:
            batch_topological_graph_matrix = None

        actual_problem_solving_step_indicator = actual_problem_solving_step_indicator.view(-1)
        caption_steps = caption_steps.view(-1, caption_steps.shape[-1])
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        # for the ground truth attention, organize like topological_graph_matrix without start/end point
        if aggregate_caption_attentions is not None:
            aggregate_caption_attentions = aggregate_caption_attentions.view(-1, aggregate_caption_attentions.shape[-1])
            _batch_aggregate_caption_attentions = aggregate_caption_attentions[actual_problem_solving_step_indicator == 1]
            batch_aggregate_caption_attentions = images.new_zeros(batch_size, batch_step_total_num,
                _batch_aggregate_caption_attentions.shape[-1])
            batch_aggregate_caption_attentions[:, :actual_total_step_num] = _batch_aggregate_caption_attentions
        else:
            _batch_aggregate_caption_attentions = None

        if scale_down_aggregate_attention_maps is not None:
            scale_down_aggregate_attention_maps = scale_down_aggregate_attention_maps.view(-1, scale_down_aggregate_attention_maps.shape[-1])
            _batch_scale_down_aggregate_attention_maps = scale_down_aggregate_attention_maps[actual_problem_solving_step_indicator == 1]
            batch_scale_down_aggregate_attention_maps = images.new_zeros(batch_size, batch_step_total_num,
                _batch_scale_down_aggregate_attention_maps.shape[-1])
            batch_scale_down_aggregate_attention_maps[:, :actual_total_step_num] = _batch_scale_down_aggregate_attention_maps
        else:
            _batch_scale_down_aggregate_attention_maps = None

        selected_caption_steps = caption_steps[actual_problem_solving_step_indicator == 1]
        selected_images = images[actual_problem_solving_step_indicator == 1]

        total_step_num = selected_caption_steps.shape[0]
        cap_emb_mask = (selected_caption_steps > 0).float()

        goal_feature = self.goal_enc(goal)
        cap_feature = self.cap_enc(selected_caption_steps)
        img_feature = self.img_enc(selected_images)

        # for loop for train with the batch_size of task
        goal_summary_feature = goal_feature[:, 0]
        # edge_matrix = actual_problem_solving_step_indicator.new_zeros((total_step_num + 2, total_step_num + 2))
        # edge_matrix[0, 2:] = 1
        # edge_matrix[2:, 1] = 1
        edge_matrix = actual_problem_solving_step_indicator.new_ones((total_step_num + 2, total_step_num + 2))
        edge_matrix[:, 0] = 0
        edge_matrix[0, 1] = 0
        edge_matrix[1, 1] = 1
        edge_matrix[1, 2:] = 0
        for idx in range(total_step_num):
            edge_matrix[idx + 2, idx + 2] = 0
        edge_index, _ = dense_to_sparse(edge_matrix)

        # go over the different goal
        _batch_image_attention_weights = []
        _batch_image_adjacency_matrix = []
        _batch_caption_attention_weights = []
        _batch_caption_adjacency_matrix = []
        for idx in range(batch_size):
            curr_goal_summary_feature = goal_summary_feature[idx].unsqueeze(0)
            image_attention_weights_list, image_adjacency_matrix_list, \
            caption_attention_weights_list, caption_adjacency_matrix_list = self.graph_enc(curr_goal_summary_feature, cap_feature, img_feature, cap_emb_mask, edge_index)

            image_attention_weights = torch.stack(image_attention_weights_list)
            image_adjacency_matrix = torch.stack(image_adjacency_matrix_list)
            caption_attention_weights = torch.stack(caption_attention_weights_list)
            caption_adjacency_matrix = torch.stack(caption_adjacency_matrix_list)

            _batch_image_attention_weights.append(image_attention_weights)
            _batch_image_adjacency_matrix.append(image_adjacency_matrix)
            _batch_caption_attention_weights.append(caption_attention_weights)
            _batch_caption_adjacency_matrix.append(caption_adjacency_matrix)

        _batch_image_attention_weights = torch.stack(_batch_image_attention_weights)
        _batch_image_adjacency_matrix = torch.stack(_batch_image_adjacency_matrix)
        _batch_caption_attention_weights = torch.stack(_batch_caption_attention_weights)
        _batch_caption_adjacency_matrix = torch.stack(_batch_caption_adjacency_matrix)

        # construct the padding result for data parallel
        batch_image_attention_weights = images.new_zeros(
            _batch_image_attention_weights.shape[0], _batch_image_attention_weights.shape[1],
            batch_step_total_num, _batch_image_attention_weights.shape[3])
        batch_image_adjacency_matrix = images.new_zeros(
            _batch_image_adjacency_matrix.shape[0], _batch_image_adjacency_matrix.shape[1],
            batch_step_total_num, batch_step_total_num)
        batch_caption_attention_weights = images.new_zeros(
            _batch_caption_attention_weights.shape[0], _batch_caption_attention_weights.shape[1],
            batch_step_total_num, _batch_caption_attention_weights.shape[3])
        batch_caption_adjacency_matrix = images.new_zeros(
            _batch_caption_adjacency_matrix.shape[0], _batch_caption_adjacency_matrix.shape[1],
            batch_step_total_num, batch_step_total_num)

        batch_image_attention_weights[:, :, :_batch_image_attention_weights.shape[2]] = _batch_image_attention_weights
        batch_image_adjacency_matrix[:, :, :_batch_image_adjacency_matrix.shape[2], :_batch_image_adjacency_matrix.shape[2]] = _batch_image_adjacency_matrix
        batch_caption_attention_weights[:, :, :_batch_caption_attention_weights.shape[2]] = _batch_caption_attention_weights
        batch_caption_adjacency_matrix[:, :, :_batch_caption_adjacency_matrix.shape[2], :_batch_caption_adjacency_matrix.shape[2]] = _batch_caption_adjacency_matrix

        loss = self.baseline_loss(_batch_image_adjacency_matrix, _batch_caption_adjacency_matrix, cap_emb_mask,
                                  batch_topological_graph_matrix[:, :actual_total_step_num + 2, :actual_total_step_num + 2],
                                  _batch_image_attention_weights, _batch_caption_attention_weights,
                                  _batch_scale_down_aggregate_attention_maps, _batch_aggregate_caption_attentions)

        data = {
            "image_attention_weights": batch_image_attention_weights,
            "image_adjacency_matrix": batch_image_adjacency_matrix,
            "caption_attention_weights": batch_caption_attention_weights,
            "caption_adjacency_matrix": batch_caption_adjacency_matrix,
            "topological_graph_matrix": batch_topological_graph_matrix,
            "aggregate_caption_attentions": batch_aggregate_caption_attentions,
            "scale_down_aggregate_attention_maps": batch_scale_down_aggregate_attention_maps,
            "actual_total_step_num": batch_topological_graph_matrix.new_ones(batch_size) * actual_total_step_num,
            "loss": loss,
        }

        return data


    def inference(self, goal, caption_steps, images, actual_problem_solving_step_indicator,
                  aggregate_caption_attentions=None, scale_down_aggregate_attention_maps=None,
                  topological_graph_matrix=None):
        batch_size = goal.shape[0]
        batch_step_num = actual_problem_solving_step_indicator.sum(-1)
        actual_total_step_num = batch_step_num.sum()
        # batch_step_total_num = actual_total_step_num + padding num
        batch_step_total_num = actual_problem_solving_step_indicator.shape[0] * actual_problem_solving_step_indicator.shape[1]
        # construct the topological_graph_matrix for the current batch
        if topological_graph_matrix is not None:
            batch_topological_graph_matrix = topological_graph_matrix.new_zeros(batch_size, batch_step_total_num, batch_step_total_num)
            start_idx = 2
            for idx in range(batch_size):
                batch_topological_graph_matrix[idx, :2, :2] = topological_graph_matrix[idx, :2, :2]
                batch_topological_graph_matrix[idx, start_idx:start_idx + batch_step_num[idx], :2] = \
                    topological_graph_matrix[idx, 2:2 + batch_step_num[idx], :2]
                batch_topological_graph_matrix[idx, :2, start_idx:start_idx + batch_step_num[idx]] = \
                    topological_graph_matrix[idx, :2, 2:2 + batch_step_num[idx]]
                batch_topological_graph_matrix[idx, start_idx:start_idx + batch_step_num[idx], start_idx:start_idx + batch_step_num[idx]] = \
                    topological_graph_matrix[idx, 2:2 + batch_step_num[idx], 2:2 + batch_step_num[idx]]
                start_idx += batch_step_num[idx]
        else:
            batch_topological_graph_matrix = None

        actual_problem_solving_step_indicator = actual_problem_solving_step_indicator.view(-1)
        caption_steps = caption_steps.view(-1, caption_steps.shape[-1])
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        # for the ground truth attention, organize like topological_graph_matrix without start/end point
        if aggregate_caption_attentions is not None:
            aggregate_caption_attentions = aggregate_caption_attentions.view(-1, aggregate_caption_attentions.shape[-1])
            _batch_aggregate_caption_attentions = aggregate_caption_attentions[actual_problem_solving_step_indicator == 1]
            batch_aggregate_caption_attentions = images.new_zeros(batch_size, batch_step_total_num,
                _batch_aggregate_caption_attentions.shape[-1])
            batch_aggregate_caption_attentions[:, :actual_total_step_num] = _batch_aggregate_caption_attentions
        else:
            _batch_aggregate_caption_attentions = None

        if scale_down_aggregate_attention_maps is not None:
            scale_down_aggregate_attention_maps = scale_down_aggregate_attention_maps.view(-1, scale_down_aggregate_attention_maps.shape[-1])
            _batch_scale_down_aggregate_attention_maps = scale_down_aggregate_attention_maps[actual_problem_solving_step_indicator == 1]
            batch_scale_down_aggregate_attention_maps = images.new_zeros(batch_size, batch_step_total_num,
                _batch_scale_down_aggregate_attention_maps.shape[-1])
            batch_scale_down_aggregate_attention_maps[:, :actual_total_step_num] = _batch_scale_down_aggregate_attention_maps
        else:
            _batch_scale_down_aggregate_attention_maps = None

        selected_caption_steps = caption_steps[actual_problem_solving_step_indicator == 1]
        selected_images = images[actual_problem_solving_step_indicator == 1]

        total_step_num = selected_caption_steps.shape[0]
        cap_emb_mask = (selected_caption_steps > 0).float()

        goal_feature = self.goal_enc(goal)
        cap_feature = self.cap_enc(selected_caption_steps)
        img_feature = self.img_enc(selected_images)

        # for loop for train with the batch_size of task
        goal_summary_feature = goal_feature[:, 0]
        # edge_matrix = actual_problem_solving_step_indicator.new_zeros((total_step_num + 2, total_step_num + 2))
        # edge_matrix[0, 2:] = 1
        # edge_matrix[2:, 1] = 1
        edge_matrix = actual_problem_solving_step_indicator.new_ones((total_step_num + 2, total_step_num + 2))
        edge_matrix[:, 0] = 0
        edge_matrix[0, 1] = 0
        edge_matrix[1, 1] = 1
        edge_matrix[1, 2:] = 0
        for idx in range(total_step_num):
            edge_matrix[idx + 2, idx + 2] = 0
        edge_index, _ = dense_to_sparse(edge_matrix)

        # go over the different goal
        _batch_image_attention_weights = []
        _batch_image_adjacency_matrix = []
        _batch_caption_attention_weights = []
        _batch_caption_adjacency_matrix = []
        for idx in range(batch_size):
            curr_goal_summary_feature = goal_summary_feature[idx].unsqueeze(0)
            image_attention_weights_list, image_adjacency_matrix_list, \
            caption_attention_weights_list, caption_adjacency_matrix_list = self.graph_enc(curr_goal_summary_feature, cap_feature, img_feature, cap_emb_mask, edge_index)

            image_attention_weights = torch.stack(image_attention_weights_list)
            image_adjacency_matrix = torch.stack(image_adjacency_matrix_list)
            caption_attention_weights = torch.stack(caption_attention_weights_list)
            caption_adjacency_matrix = torch.stack(caption_adjacency_matrix_list)

            _batch_image_attention_weights.append(image_attention_weights)
            _batch_image_adjacency_matrix.append(image_adjacency_matrix)
            _batch_caption_attention_weights.append(caption_attention_weights)
            _batch_caption_adjacency_matrix.append(caption_adjacency_matrix)

        _batch_image_attention_weights = torch.stack(_batch_image_attention_weights)
        _batch_image_adjacency_matrix = torch.stack(_batch_image_adjacency_matrix)
        _batch_caption_attention_weights = torch.stack(_batch_caption_attention_weights)
        _batch_caption_adjacency_matrix = torch.stack(_batch_caption_adjacency_matrix)

        # construct the padding result for data parallel
        batch_image_attention_weights = images.new_zeros(
            _batch_image_attention_weights.shape[0], _batch_image_attention_weights.shape[1],
            batch_step_total_num, _batch_image_attention_weights.shape[3])
        batch_image_adjacency_matrix = images.new_zeros(
            _batch_image_adjacency_matrix.shape[0], _batch_image_adjacency_matrix.shape[1],
            batch_step_total_num, batch_step_total_num)
        batch_caption_attention_weights = images.new_zeros(
            _batch_caption_attention_weights.shape[0], _batch_caption_attention_weights.shape[1],
            batch_step_total_num, _batch_caption_attention_weights.shape[3])
        batch_caption_adjacency_matrix = images.new_zeros(
            _batch_caption_adjacency_matrix.shape[0], _batch_caption_adjacency_matrix.shape[1],
            batch_step_total_num, batch_step_total_num)

        batch_image_attention_weights[:, :, :_batch_image_attention_weights.shape[2]] = _batch_image_attention_weights
        batch_image_adjacency_matrix[:, :, :_batch_image_adjacency_matrix.shape[2], :_batch_image_adjacency_matrix.shape[2]] = _batch_image_adjacency_matrix
        batch_caption_attention_weights[:, :, :_batch_caption_attention_weights.shape[2]] = _batch_caption_attention_weights
        batch_caption_adjacency_matrix[:, :, :_batch_caption_adjacency_matrix.shape[2], :_batch_caption_adjacency_matrix.shape[2]] = _batch_caption_adjacency_matrix

        loss = self.baseline_loss(_batch_image_adjacency_matrix, _batch_caption_adjacency_matrix, cap_emb_mask,
                                  batch_topological_graph_matrix[:, :actual_total_step_num + 2, :actual_total_step_num + 2],
                                  _batch_image_attention_weights, _batch_caption_attention_weights,
                                  _batch_scale_down_aggregate_attention_maps, _batch_aggregate_caption_attentions)

        data = {
            "image_attention_weights": batch_image_attention_weights,
            "image_adjacency_matrix": batch_image_adjacency_matrix,
            "caption_attention_weights": batch_caption_attention_weights,
            "caption_adjacency_matrix": batch_caption_adjacency_matrix,
            "topological_graph_matrix": batch_topological_graph_matrix,
            "aggregate_caption_attentions": batch_aggregate_caption_attentions,
            "scale_down_aggregate_attention_maps": batch_scale_down_aggregate_attention_maps,
            "actual_total_step_num": batch_topological_graph_matrix.new_ones(batch_size) * actual_total_step_num,
            "loss": loss,
        }

        return data