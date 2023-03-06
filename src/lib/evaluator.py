"""Evaluation"""
from __future__ import print_function

import json
import os
import time
import torch
import numpy as np
from torch import Tensor

from collections import OrderedDict
import logging
from tqdm import tqdm
from lib.manager import BaselineManager
from lib.datasets import visualhow

from transformers import BertTokenizer

import scipy.stats
import sys

from sklearn import metrics
import pickle
from scipy.stats import spearmanr,pearsonr

import tempfile
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

logger = logging.getLogger(__name__)

# evaluation metrics
def cal_cc_score(pred, gt):
    eps = 1e-15
    pred = pred + eps
    gt = gt + eps
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)

    if np.std(gt.reshape(-1)) <= 1e-8 or np.std(pred.reshape(-1)) <= 1e-8:
        cc_score = 1
    else:
        cc_score = np.corrcoef(pred.reshape(-1), gt.reshape(-1))[0][1]

    return cc_score

def cal_sim_score(pred, gt):
    eps = 1e-15
    pred = pred + eps
    gt = gt + eps
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)

    sim_score = np.sum(np.minimum(pred, gt))

    return sim_score

def cal_kld_score(pred, gt):
    eps = 1e-15
    pred = pred + eps
    gt = gt + eps
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)
    kl_score = gt * np.log(eps + gt / (pred + eps))
    kl_score = np.sum(kl_score)

    return kl_score


class Evaluator(object):
    def __init__(self, opt, mode='val'):
        self.opt = opt
        self.eval_name = self.opt.eval_name
        self.prediction_file = os.path.join(self.eval_name, 'prediction_{}.json'.format(mode))
        self.detailed_evaluation_file = os.path.join(self.eval_name, 'detailed_evaluation_{}.json'.format(mode))
        self.data = None
        self.sample_num = 0
        self.evaluation_layer = -1
        self.similarity_function = cal_sim_score
        self.is_step_aspect = True


    def load_data(self, data):
        self.data = data
        self.sample_num = len(self.data)
        for key, value in data.items():
            value["metric"] = {}

    def _message_propagation(self, att_score, topological_graph_matrix, predict_topological_graph_matrix):
        eps = 1e-15
        GP = topological_graph_matrix * predict_topological_graph_matrix

        out_degree = GP.sum(1, keepdims=True) / (predict_topological_graph_matrix.sum(1, keepdims=True) + eps)

        information_dist = GP / (topological_graph_matrix.sum(1, keepdims=True) + eps)

        in_degree = GP.sum(0, keepdims=True) / (predict_topological_graph_matrix.sum(0, keepdims=True) + eps)

        M = out_degree * information_dist * in_degree

        # remove using too much propagation, at this time we use 1 time propagation
        # for idx in range(topological_graph_matrix[0, 2:].sum() + 1):
        #     att_score = M.T @ att_score
        #
        # score = att_score[1] / (topological_graph_matrix[0, 2:].sum() + 1)

        att_score = M.T @ att_score
        score = att_score.sum() / (topological_graph_matrix[0, 2:].sum() + 1)

        return score

    def _message_propagation_in_degree(self, att_score, topological_graph_matrix, predict_topological_graph_matrix):
        eps = 1e-15
        GP = topological_graph_matrix * predict_topological_graph_matrix

        information_dist = GP / (topological_graph_matrix.sum(1, keepdims=True) + eps)

        in_degree = GP.sum(0, keepdims=True) / (predict_topological_graph_matrix.sum(0, keepdims=True) + eps)

        M = information_dist * in_degree

        # remove using too much propagation, at this time we use 1 time propagation
        # for idx in range(topological_graph_matrix[0, 2:].sum() + 1):
        #     att_score = M.T @ att_score
        #
        # score = att_score[1] / (topological_graph_matrix[0, 2:].sum() + 1)

        att_score = M.T @ att_score
        score = att_score.sum() / (topological_graph_matrix[0, 2:].sum() + 1)

        return score

    def _message_propagation_out_degree(self, att_score, topological_graph_matrix, predict_topological_graph_matrix):
        eps = 1e-15
        GP = topological_graph_matrix * predict_topological_graph_matrix

        out_degree = GP.sum(1, keepdims=True) / (predict_topological_graph_matrix.sum(1, keepdims=True) + eps)

        information_dist = GP / (topological_graph_matrix.sum(1, keepdims=True) + eps)

        M = out_degree * information_dist

        # remove using too much propagation, at this time we use 1 time propagation
        # for idx in range(topological_graph_matrix[0, 2:].sum() + 1):
        #     att_score = M.T @ att_score
        #
        # score = att_score[1] / (topological_graph_matrix[0, 2:].sum() + 1)

        att_score = M.T @ att_score
        score = att_score.sum() / (topological_graph_matrix[0, 2:].sum() + 1)

        return score


    def _measure_recall_task_aspect(self):
        # the problem-solving step connect to the task is encoded in ground_truth_graph[0, :2]

        # for image domain
        image_rank = []
        for key, value in self.data.items():
            image_graph = value["image_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            step_num = ground_truth_graph[0, 2:].sum()
            image_prediction = image_graph[0, 2:]
            sorted_index = np.argsort(image_prediction)[::-1]
            candidate_position = np.where(ground_truth_graph[0, 2:] == 1)[0]
            problem_solving_predict_idx = np.concatenate([np.where(sorted_index == i) for i in list(candidate_position)])
            image_rank.append(problem_solving_predict_idx.max() - (step_num - 1))
        image_rank = np.array(image_rank)

        # for caption domain
        caption_rank = []
        for key, value in self.data.items():
            caption_graph = value["caption_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            step_num = ground_truth_graph[0, 2:].sum()
            caption_prediction = caption_graph[0, 2:]
            sorted_index = np.argsort(caption_prediction)[::-1]
            candidate_position = np.where(ground_truth_graph[0, 2:] == 1)[0]
            problem_solving_predict_idx = np.concatenate([np.where(sorted_index == i) for i in list(candidate_position)])
            caption_rank.append(problem_solving_predict_idx.max() - (step_num - 1))
        caption_rank = np.array(caption_rank)

        # Compute metrics
        ir1 = 100.0 * len(np.where(image_rank < 1)[0]) / len(image_rank)
        ir5 = 100.0 * len(np.where(image_rank < 5)[0]) / len(image_rank)
        ir10 = 100.0 * len(np.where(image_rank < 10)[0]) / len(image_rank)
        imedr = np.floor(np.median(image_rank)) + 1
        imeanr = image_rank.mean() + 1
        imrr = (1 / (image_rank + 1)).mean()

        cr1 = 100.0 * len(np.where(caption_rank < 1)[0]) / len(caption_rank)
        cr5 = 100.0 * len(np.where(caption_rank < 5)[0]) / len(caption_rank)
        cr10 = 100.0 * len(np.where(caption_rank < 10)[0]) / len(caption_rank)
        cmedr = np.floor(np.median(caption_rank)) + 1
        cmeanr = caption_rank.mean() + 1
        cmrr = (1 / (caption_rank + 1)).mean()

        return (ir1, ir5, ir10, imedr, imeanr, imrr), (cr1, cr5, cr10, cmedr, cmeanr, cmrr)

    def _measure_recall_step_aspect(self):
        # the problem-solving step connect to the task is encoded in ground_truth_graph[0, :2]

        # for image domain
        image_rank = []
        for key, value in self.data.items():
            image_graph = value["image_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            step_num = ground_truth_graph[0, 2:].sum()
            image_prediction = image_graph[0, 2:]
            sorted_index = np.argsort(image_prediction)[::-1]
            candidate_position = np.where(ground_truth_graph[0, 2:] == 1)[0]
            problem_solving_predict_idx = np.concatenate(
                [np.where(sorted_index == i) for i in list(candidate_position)])
            problem_solving_predict_idx = np.sort(problem_solving_predict_idx.squeeze())
            problem_solving_predict_idx_remove_bias = problem_solving_predict_idx - np.arange(
                problem_solving_predict_idx.shape[0])
            image_rank.append(problem_solving_predict_idx_remove_bias)
            value["metric"]["image_rank"] = problem_solving_predict_idx_remove_bias
        image_rank = np.concatenate(image_rank)

        # for caption domain
        caption_rank = []
        for key, value in self.data.items():
            caption_graph = value["caption_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            step_num = ground_truth_graph[0, 2:].sum()
            caption_prediction = caption_graph[0, 2:]
            sorted_index = np.argsort(caption_prediction)[::-1]
            candidate_position = np.where(ground_truth_graph[0, 2:] == 1)[0]
            problem_solving_predict_idx = np.concatenate(
                [np.where(sorted_index == i) for i in list(candidate_position)])
            problem_solving_predict_idx = np.sort(problem_solving_predict_idx.squeeze())
            problem_solving_predict_idx_remove_bias = problem_solving_predict_idx - np.arange(
                problem_solving_predict_idx.shape[0])
            caption_rank.append(problem_solving_predict_idx_remove_bias)
            value["metric"]["caption_rank"] = problem_solving_predict_idx_remove_bias
        caption_rank = np.concatenate(caption_rank)

        # Compute metrics
        ir1 = 100.0 * len(np.where(image_rank < 1)[0]) / len(image_rank)
        ir5 = 100.0 * len(np.where(image_rank < 5)[0]) / len(image_rank)
        ir10 = 100.0 * len(np.where(image_rank < 10)[0]) / len(image_rank)
        imedr = np.floor(np.median(image_rank)) + 1
        imeanr = image_rank.mean() + 1
        imrr = (1 / (image_rank + 1)).mean()

        cr1 = 100.0 * len(np.where(caption_rank < 1)[0]) / len(caption_rank)
        cr5 = 100.0 * len(np.where(caption_rank < 5)[0]) / len(caption_rank)
        cr10 = 100.0 * len(np.where(caption_rank < 10)[0]) / len(caption_rank)
        cmedr = np.floor(np.median(caption_rank)) + 1
        cmeanr = caption_rank.mean() + 1
        cmrr = (1 / (caption_rank + 1)).mean()

        return (ir1, ir5, ir10, imedr, imeanr, imrr), (cr1, cr5, cr10, cmedr, cmeanr, cmrr)

    def _measure_graph(self):
        # for image domain
        image_auc = []
        image_iou = {_: [] for _ in ["0.25", "0.5", "0.75"]}
        for key, value in self.data.items():
            image_graph = value["image_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth_graph.reshape(-1), image_graph.reshape(-1), pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            image_auc.append(auc_value)
            value["metric"]["image_graph"] = {
                "auc": auc_value,
            }
            for iou_key, iou_value in image_iou.items():
                image_graph_hard = image_graph >= float(iou_key)
                iou = (ground_truth_graph * image_graph_hard).sum() / ((ground_truth_graph + image_graph_hard) > 0).sum()
                iou_value.append(iou)
                value["metric"]["image_graph"][iou_key] = iou

        image_auc = np.array(image_auc)
        for key, value in image_iou.items():
            image_iou[key] = np.array(value)

        # for caption domain
        caption_auc = []
        caption_iou = {_: [] for _ in ["0.25", "0.5", "0.75"]}
        for key, value in self.data.items():
            caption_graph = value["caption_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth_graph.reshape(-1), caption_graph.reshape(-1),
                                                     pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            caption_auc.append(auc_value)
            value["metric"]["caption_graph"] = {
                "auc": auc_value,
            }

            for iou_key, iou_value in caption_iou.items():
                caption_graph_hard = caption_graph >= float(iou_key)
                iou = (ground_truth_graph * caption_graph_hard).sum() / (
                            (ground_truth_graph + caption_graph_hard) > 0).sum()
                iou_value.append(iou)
                value["metric"]["caption_graph"][iou_key] = iou
        caption_auc = np.array(caption_auc)
        for key, value in caption_iou.items():
            caption_iou[key] = np.array(value)

        caption_auc = np.array(caption_auc)
        for key, value in caption_iou.items():
            caption_iou[key] = np.array(value)

        iauc = image_auc.mean()
        cauc = caption_auc.mean()
        iiou = {}
        for key, value in image_iou.items():
            iiou[key] = value.mean()
        ciou = {}
        for key, value in caption_iou.items():
            ciou[key] = value.mean()

        return (iauc, iiou), (cauc, ciou)

    def _measure_graph_for_gt_steps_separate_goal(self):
        # for image domain
        image_auc = []
        image_aupr = []
        image_iou = {_: [] for _ in ["0.25", "0.5", "0.75"]}
        for key, value in self.data.items():
            image_graph = value["image_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            # consider end point instead of start point, since retrieval metric consider it
            gt_step_idx = np.where(ground_truth_graph[0] !=0)[0]
            gt_step_with_end_idx = np.concatenate([[1], gt_step_idx])
            selection_mask = np.zeros(ground_truth_graph.shape)
            selection_mask[gt_step_idx] += 1
            selection_mask[:, gt_step_with_end_idx] += 1
            selection_mask = (selection_mask >= 2)
            for idx in gt_step_idx:
                selection_mask[idx, idx] = 0
            selection_mask[0, gt_step_idx] = 1
            ground_truth_graph = ground_truth_graph[selection_mask]
            image_graph = image_graph[selection_mask]
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth_graph.reshape(-1),
                                                     image_graph.reshape(-1), pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            image_auc.append(auc_value)
            value["metric"]["image_graph"] = {
                "auc": auc_value,
            }

            precision, recall, thresholds = metrics.precision_recall_curve(ground_truth_graph.reshape(-1),
                                                                           image_graph.reshape(-1), pos_label=1)
            aupr_value = metrics.auc(recall, precision)
            image_aupr.append(aupr_value)
            value["metric"]["image_graph"]["aupr"] = aupr_value

            for iou_key, iou_value in image_iou.items():
                image_graph_hard = image_graph >= float(iou_key)
                iou = (ground_truth_graph * image_graph_hard).sum() / ((ground_truth_graph + image_graph_hard) > 0).sum()
                iou_value.append(iou)
                value["metric"]["image_graph"][iou_key] = iou

        image_auc = np.array(image_auc)
        image_aupr = np.array(image_aupr)
        for key, value in image_iou.items():
            image_iou[key] = np.array(value)

        # for caption domain
        caption_auc = []
        caption_aupr = []
        caption_iou = {_: [] for _ in ["0.25", "0.5", "0.75"]}
        for key, value in self.data.items():
            caption_graph = value["caption_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            # consider end point instead of start point, since retrieval metric consider it
            gt_step_idx = np.where(ground_truth_graph[0] != 0)[0]
            gt_step_with_end_idx = np.concatenate([[1], gt_step_idx])
            selection_mask = np.zeros(ground_truth_graph.shape)
            selection_mask[gt_step_idx] += 1
            selection_mask[:, gt_step_with_end_idx] += 1
            selection_mask = (selection_mask >= 2)
            for idx in gt_step_idx:
                selection_mask[idx, idx] = 0
            selection_mask[0, gt_step_idx] = 1
            ground_truth_graph = ground_truth_graph[selection_mask]
            caption_graph = caption_graph[selection_mask]
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth_graph.reshape(-1), caption_graph.reshape(-1),
                                                     pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            caption_auc.append(auc_value)
            value["metric"]["caption_graph"] = {
                "auc": auc_value,
            }

            precision, recall, thresholds = metrics.precision_recall_curve(ground_truth_graph.reshape(-1),
                                                                           caption_graph.reshape(-1), pos_label=1)
            aupr_value = metrics.auc(recall, precision)
            caption_aupr.append(aupr_value)
            value["metric"]["caption_graph"]["aupr"] = aupr_value

            for iou_key, iou_value in caption_iou.items():
                caption_graph_hard = caption_graph >= float(iou_key)
                iou = (ground_truth_graph * caption_graph_hard).sum() / (
                            (ground_truth_graph + caption_graph_hard) > 0).sum()
                iou_value.append(iou)
                value["metric"]["caption_graph"][iou_key] = iou
        caption_auc = np.array(caption_auc)
        caption_aupr = np.array(caption_aupr)
        for key, value in caption_iou.items():
            caption_iou[key] = np.array(value)

        iauc = image_auc.mean()
        cauc = caption_auc.mean()
        iaupr = image_aupr.mean()
        caupr = caption_aupr.mean()
        iiou = {}
        for key, value in image_iou.items():
            iiou[key] = value.mean()
        ciou = {}
        for key, value in caption_iou.items():
            ciou[key] = value.mean()

        return (iauc, iaupr, iiou), (cauc, caupr, ciou)


    def _measure_graph_for_gt_steps(self):
        # for image domain
        image_ground_truth_graph = []
        image_graph_prediction = []
        image_iou = {_: [] for _ in ["0.25", "0.5", "0.75"]}
        for key, value in self.data.items():
            image_graph = value["image_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            # consider end point instead of start point, since retrieval metric consider it
            gt_step_idx = np.where(ground_truth_graph[0] != 0)[0]
            gt_step_with_end_idx = np.concatenate([[1], gt_step_idx])
            selection_mask = np.zeros(ground_truth_graph.shape)
            selection_mask[gt_step_idx] += 1
            selection_mask[:, gt_step_with_end_idx] += 1
            selection_mask = (selection_mask >= 2)
            for idx in gt_step_idx:
                selection_mask[idx, idx] = 0
            selection_mask[0, gt_step_idx] = 1
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth_graph.reshape(-1),
                                                     image_graph.reshape(-1), pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            value["metric"]["image_graph"] = {
                "auc": auc_value,
            }

            precision, recall, thresholds = metrics.precision_recall_curve(ground_truth_graph.reshape(-1),
                                                                           image_graph.reshape(-1), pos_label=1)
            aupr_value = metrics.auc(recall, precision)
            value["metric"]["image_graph"]["aupr"] = aupr_value

            ground_truth_graph = ground_truth_graph[selection_mask]
            image_graph = image_graph[selection_mask]
            image_ground_truth_graph.append(ground_truth_graph)
            image_graph_prediction.append(image_graph)

            for iou_key, iou_value in image_iou.items():
                image_graph_hard = image_graph >= float(iou_key)
                iou = (ground_truth_graph * image_graph_hard).sum() / (
                            (ground_truth_graph + image_graph_hard) > 0).sum()
                iou_value.append(iou)
                value["metric"]["image_graph"][iou_key] = iou

        # for caption domain
        caption_ground_truth_graph = []
        caption_graph_prediction = []
        caption_iou = {_: [] for _ in ["0.25", "0.5", "0.75"]}
        for key, value in self.data.items():
            caption_graph = value["caption_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            # consider end point instead of start point, since retrieval metric consider it
            gt_step_idx = np.where(ground_truth_graph[0] != 0)[0]
            gt_step_with_end_idx = np.concatenate([[1], gt_step_idx])
            selection_mask = np.zeros(ground_truth_graph.shape)
            selection_mask[gt_step_idx] += 1
            selection_mask[:, gt_step_with_end_idx] += 1
            selection_mask = (selection_mask >= 2)
            for idx in gt_step_idx:
                selection_mask[idx, idx] = 0
            selection_mask[0, gt_step_idx] = 1
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth_graph.reshape(-1), caption_graph.reshape(-1),
                                                     pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            value["metric"]["caption_graph"] = {
                "auc": auc_value,
            }

            precision, recall, thresholds = metrics.precision_recall_curve(ground_truth_graph.reshape(-1),
                                                                           caption_graph.reshape(-1), pos_label=1)
            aupr_value = metrics.auc(recall, precision)
            value["metric"]["caption_graph"]["aupr"] = aupr_value

            ground_truth_graph = ground_truth_graph[selection_mask]
            caption_graph = caption_graph[selection_mask]
            caption_ground_truth_graph.append(ground_truth_graph)
            caption_graph_prediction.append(caption_graph)

            for iou_key, iou_value in caption_iou.items():
                caption_graph_hard = caption_graph >= float(iou_key)
                iou = (ground_truth_graph * caption_graph_hard).sum() / (
                        (ground_truth_graph + caption_graph_hard) > 0).sum()
                iou_value.append(iou)
                value["metric"]["caption_graph"][iou_key] = iou

        # auc
        image_ground_truth_graph = np.concatenate(image_ground_truth_graph)
        image_graph_prediction = np.concatenate(image_graph_prediction)
        fpr, tpr, thresholds = metrics.roc_curve(image_ground_truth_graph.reshape(-1),
                                                 image_graph_prediction.reshape(-1),
                                                 pos_label=1)
        iauc = metrics.auc(fpr, tpr)

        caption_ground_truth_graph = np.concatenate(caption_ground_truth_graph)
        caption_graph_prediction = np.concatenate(caption_graph_prediction)
        fpr, tpr, thresholds = metrics.roc_curve(caption_ground_truth_graph.reshape(-1),
                                                 caption_graph_prediction.reshape(-1),
                                                 pos_label=1)
        cauc = metrics.auc(fpr, tpr)

        # AUPR
        precision, recall, thresholds = metrics.precision_recall_curve(image_ground_truth_graph.reshape(-1),
                                                              image_graph_prediction.reshape(-1),
                                                              pos_label=1)
        iaupr = metrics.auc(recall, precision)

        precision, recall, thresholds = metrics.precision_recall_curve(caption_ground_truth_graph.reshape(-1),
                                                              caption_graph_prediction.reshape(-1),
                                                              pos_label=1)
        caupr = metrics.auc(recall, precision)


        iiou = {}
        for key, value in image_iou.items():
            image_graph_hard = image_graph_prediction >= float(key)
            iiou[key] = (image_ground_truth_graph * image_graph_hard).sum() / (
                    (image_ground_truth_graph + image_graph_hard) > 0).sum()
        ciou = {}
        for key, value in caption_iou.items():
            caption_graph_hard = caption_graph_prediction >= float(key)
            ciou[key] = (caption_ground_truth_graph * caption_graph_hard).sum() / (
                    (caption_ground_truth_graph + caption_graph_hard) > 0).sum()

        return (iauc, iaupr, iiou), (cauc, caupr, ciou)

    def _measure_attention(self):
        # attention evaluation
        image_eval_score = dict()
        for metric in ['CC', 'SIM', 'KLD', 'Spearman']:
            image_eval_score[metric] = []
        image_eval_score["id"] = []

        for key, value in self.data.items():
            ground_truth_graph = value["ground_truth_graph"]
            image_attention_weight = value["image_attention_weight"][self.evaluation_layer]
            ground_truth_image_attention_weight = value["ground_truth_image_attention_weight"]
            cur_image_attention_weight = image_attention_weight[ground_truth_graph[0, 2:] == 1]
            cur_ground_truth_image_attention_weight = ground_truth_image_attention_weight[ground_truth_graph[0, 2:] == 1]
            value["metric"]["image_attention"] = {}

            for idx in range(cur_image_attention_weight.shape[0]):
                cur_gt = cur_ground_truth_image_attention_weight[idx]
                cur_pred = cur_image_attention_weight[idx]

                cur_cc = cal_cc_score(cur_pred, cur_gt)
                cur_kld = cal_kld_score(cur_pred, cur_gt)
                cur_sim = cal_sim_score(cur_pred, cur_gt)
                cur_spearman = spearmanr(cur_pred, cur_gt)[0]

                image_eval_score['CC'].append(cur_cc if not np.isnan(cur_cc) else 0)
                image_eval_score['SIM'].append(cur_sim if not np.isnan(cur_sim) else 0)
                image_eval_score['KLD'].append(cur_kld if not np.isnan(cur_kld) else 10)
                image_eval_score['Spearman'].append(cur_spearman if not np.isnan(cur_spearman) else 0)
                image_eval_score["id"].append("{}_{}".format(key, idx))
                value["metric"]["image_attention"].setdefault("CC", []).append(cur_cc if not np.isnan(cur_cc) else 0)
                value["metric"]["image_attention"].setdefault("SIM", []).append(cur_sim if not np.isnan(cur_sim) else 0)
                value["metric"]["image_attention"].setdefault("KLD", []).append(cur_kld if not np.isnan(cur_kld) else 10)
                value["metric"]["image_attention"].setdefault("Spearman", []).append(cur_spearman if not np.isnan(cur_spearman) else 0)
                value["metric"]["image_attention"].setdefault("id", []).append("{}_{}".format(key, idx))

        # attention evaluation
        caption_eval_score = dict()
        for metric in ['CC', 'SIM', 'KLD', 'Spearman']:
            caption_eval_score[metric] = []
        caption_eval_score["id"] = []

        for key, value in self.data.items():
            ground_truth_graph = value["ground_truth_graph"]
            caption_attention_weight = value["caption_attention_weight"][self.evaluation_layer]
            ground_truth_caption_attention_weight = value["ground_truth_caption_attention_weight"]
            cur_caption_attention_weight = caption_attention_weight[ground_truth_graph[0, 2:] == 1]
            cur_ground_truth_caption_attention_weight = ground_truth_caption_attention_weight[ground_truth_graph[0, 2:] == 1]
            value["metric"]["caption_attention"] = {}

            for idx in range(cur_caption_attention_weight.shape[0]):
                cur_gt = cur_ground_truth_caption_attention_weight[idx]
                cur_pred = cur_caption_attention_weight[idx]

                cur_cc = cal_cc_score(cur_pred, cur_gt)
                cur_kld = cal_kld_score(cur_pred, cur_gt)
                cur_sim = cal_sim_score(cur_pred, cur_gt)
                cur_spearman = spearmanr(cur_pred, cur_gt)[0]

                caption_eval_score['CC'].append(cur_cc if not np.isnan(cur_cc) else 0)
                caption_eval_score['SIM'].append(cur_sim if not np.isnan(cur_sim) else 0)
                caption_eval_score['KLD'].append(cur_kld if not np.isnan(cur_kld) else 10)
                caption_eval_score['Spearman'].append(cur_spearman if not np.isnan(cur_spearman) else 0)
                caption_eval_score["id"].append("{}_{}".format(key, idx))

                value["metric"]["caption_attention"].setdefault("CC", []).append(cur_cc if not np.isnan(cur_cc) else 0)
                value["metric"]["caption_attention"].setdefault("SIM", []).append(cur_sim if not np.isnan(cur_sim) else 0)
                value["metric"]["caption_attention"].setdefault("KLD", []).append(cur_kld if not np.isnan(cur_kld) else 10)
                value["metric"]["caption_attention"].setdefault("Spearman", []).append(cur_spearman if not np.isnan(cur_spearman) else 0)
                value["metric"]["caption_attention"].setdefault("id", []).append("{}_{}".format(key, idx))

        return image_eval_score, caption_eval_score


    def _measure_completion(self):
        image_completion = []
        image_in_degree_completion = []
        image_out_degree_completion = []
        for key, value in self.data.items():
            image_graph = value["image_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            image_attention_weight = value["image_attention_weight"][self.evaluation_layer]
            ground_truth_image_attention_weight = value["ground_truth_image_attention_weight"]

            candidate_position = np.where(ground_truth_graph[0, 2:] == 1)[0]
            att_score = np.zeros(ground_truth_graph.shape[0])
            att_score[0] = 1
            for idx in candidate_position:
                att_score[idx + 2] = self.similarity_function(image_attention_weight[idx], ground_truth_image_attention_weight[idx])

            score = self._message_propagation(att_score, ground_truth_graph, image_graph)
            image_completion.append(score)
            value["metric"]["image_completion"] = score

            in_degree_score = self._message_propagation_in_degree(att_score, ground_truth_graph, image_graph)
            image_in_degree_completion.append(in_degree_score)
            value["metric"]["image_in_degree_completion"] = in_degree_score

            out_degree_score = self._message_propagation_out_degree(att_score, ground_truth_graph, image_graph)
            image_out_degree_completion.append(out_degree_score)
            value["metric"]["image_out_degree_completion"] = out_degree_score

        image_completion = np.array(image_completion)
        image_in_degree_completion = np.array(image_in_degree_completion)
        image_out_degree_completion = np.array(image_out_degree_completion)

        caption_completion = []
        caption_in_degree_completion = []
        caption_out_degree_completion = []
        for key, value in self.data.items():
            caption_graph = value["caption_graph"][self.evaluation_layer]
            ground_truth_graph = value["ground_truth_graph"]
            caption_attention_weight = value["caption_attention_weight"][self.evaluation_layer]
            ground_truth_caption_attention_weight = value["ground_truth_caption_attention_weight"]

            candidate_position = np.where(ground_truth_graph[0, 2:] == 1)[0]
            att_score = np.zeros(ground_truth_graph.shape[0])
            att_score[0] = 1
            for idx in candidate_position:
                att_score[idx + 2] = self.similarity_function(caption_attention_weight[idx],
                                                              ground_truth_caption_attention_weight[idx])

            score = self._message_propagation(att_score, ground_truth_graph, caption_graph)
            caption_completion.append(score)
            value["metric"]["caption_completion"] = score

            in_degree_score = self._message_propagation_in_degree(att_score, ground_truth_graph, caption_graph)
            caption_in_degree_completion.append(in_degree_score)
            value["metric"]["caption_in_degree_completion"] = in_degree_score

            out_degree_score = self._message_propagation_out_degree(att_score, ground_truth_graph, caption_graph)
            caption_out_degree_completion.append(out_degree_score)
            value["metric"]["caption_out_degree_completion"] = out_degree_score

        caption_completion = np.array(caption_completion)
        caption_in_degree_completion = np.array(caption_in_degree_completion)
        caption_out_degree_completion = np.array(caption_out_degree_completion)


        icompletion = image_completion.mean()
        ccompletion = caption_completion.mean()

        i_indegree_completion = image_in_degree_completion.mean()
        i_outdegree_completion = image_out_degree_completion.mean()
        c_indegree_completion = caption_in_degree_completion.mean()
        c_outdegree_completion = caption_out_degree_completion.mean()


        # icompletion_loss = []
        # ccompletion_loss = []
        # for key, value in self.data.items():
        #     icompletion_loss.append(value["img_completion_loss"])
        #     ccompletion_loss.append(value["cap_completion_loss"])
        # icompletion_loss = np.stack(icompletion_loss)
        # ccompletion_loss = np.stack(ccompletion_loss)
        return icompletion, i_indegree_completion, i_outdegree_completion, ccompletion, c_indegree_completion, c_outdegree_completion

    def measure(self):
        assert self.data is not None, "The data to be evaluated is None!"

        if self.is_step_aspect:
            img_recall, caption_recall = self._measure_recall_step_aspect()
        else:
            img_recall, caption_recall = self._measure_recall_task_aspect()
        # img_graph, caption_graph = self._measure_graph_for_gt_steps_separate_goal()
        img_graph, caption_graph = self._measure_graph_for_gt_steps()
        image_completion, image_in_degree_completion, image_out_degree_completion, \
        caption_completion, caption_in_degree_completion, caption_out_degree_completion = self._measure_completion()
        image_attention, caption_attention = self._measure_attention()

        rsum = img_recall[0] + img_recall[1] + img_recall[2] + caption_recall[0] + caption_recall[1] + caption_recall[2]
        cur_metrics = {
            "rsum": rsum,
            "ir1": img_recall[0],
            "ir5": img_recall[1],
            "ir10": img_recall[2],
            "imedr": img_recall[3],
            "imeanr": img_recall[4],
            "imrr": img_recall[5],
            "cr1": caption_recall[0],
            "cr5": caption_recall[1],
            "cr10": caption_recall[2],
            "cmedr": caption_recall[3],
            "cmeanr": caption_recall[4],
            "cmrr": caption_recall[5],
            "iauc": img_graph[0],
            "iaupr": img_graph[1],
            "iiou025": img_graph[2]["0.25"],
            "iiou05": img_graph[2]["0.5"],
            "iiou075": img_graph[2]["0.75"],
            "cauc": caption_graph[0],
            "caupr": caption_graph[1],
            "ciou025": caption_graph[2]["0.25"],
            "ciou05": caption_graph[2]["0.5"],
            "ciou075": caption_graph[2]["0.75"],
            "image_completion": image_completion,
            "image_in_degree_completion": image_in_degree_completion,
            "image_out_degree_completion": image_out_degree_completion,
            "caption_completion": caption_completion,
            "caption_in_degree_completion": caption_in_degree_completion,
            "caption_out_degree_completion": caption_out_degree_completion,
            "iCC": float(np.mean(image_attention['CC'])),
            "iSIM": float(np.mean(image_attention['SIM'])),
            "iKLD": float(np.mean(image_attention['KLD'])),
            "iSpearman": float(np.mean(image_attention['Spearman'])),
            "cCC": float(np.mean(caption_attention['CC'])),
            "cSIM": float(np.mean(caption_attention['SIM'])),
            "cKLD": float(np.mean(caption_attention['KLD'])),
            "cSpearman": float(np.mean(caption_attention['Spearman'])),
        }

        return cur_metrics
