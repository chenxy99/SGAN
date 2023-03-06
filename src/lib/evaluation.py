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
from lib.evaluator import Evaluator

from transformers import BertTokenizer

import scipy.stats
import sys

from sklearn import metrics
import pickle
import matplotlib.pyplot as plt

import tempfile
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, global_step=step)


def get_prediction(model, data_loader, opt, log_step=10, logging=logger.info):
    """
    Get prediction of the final graph
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # key is the task and method id
    results = {}
    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        tmp = [data_i["goal"].to('cuda'), data_i["caption_steps"].to('cuda'),
               data_i["images"].to('cuda'), data_i["aggregate_caption_attentions"].to('cuda'),
               data_i["scale_down_aggregate_attention_maps"].to('cuda'),
               data_i["actual_problem_solving_step_indicator"].to('cuda'),
               data_i["topological_graph_matrix"].to('cuda')]

        goal, caption_steps, images, aggregate_caption_attentions, scale_down_aggregate_attention_maps,\
            actual_problem_solving_step_indicator, topological_graph_matrix = tmp

        model.logger = val_logger

        output = model.inference(goal, caption_steps, images, aggregate_caption_attentions, scale_down_aggregate_attention_maps,
                    actual_problem_solving_step_indicator, topological_graph_matrix)

        for key, value in output.items():
            if isinstance(value, Tensor):
                output[key] = value.cpu().numpy()
            if key == "loss":
                for key_, value_ in output["loss"].items():
                    if isinstance(value_, Tensor):
                        output[key][key_] = value_.cpu().numpy()

        for idx in range(output["actual_total_step_num"].shape[0]):
            result = {
                "dependency_type": data_i["information"][idx]["dependency_type"],
                "information": data_i["information"][idx],
                "batch_information": data_i["information"],
                "image_graph": None,
                "caption_graph": None,
                "ground_truth_graph": None,
                "image_attention_weight": None,
                "caption_attention_weight": None,
                "ground_truth_image_attention_weight": None,
                "ground_truth_caption_attention_weight": None,
                # "img_completion_loss": None,
                # "cap_completion_loss": None,
            }
            # for the actual problem-solving steps
            actual_candidate_num = output["actual_total_step_num"][idx]
            # for the image graph and caption graph, it needs to consider the start/end nodes
            including_step_num = output["actual_total_step_num"][idx] + 2
            # predicted graph
            result["image_graph"] = output["image_adjacency_matrix"][idx, :, :including_step_num, :including_step_num]
            result["caption_graph"] = output["caption_adjacency_matrix"][idx, :, :including_step_num, :including_step_num]
            result["ground_truth_graph"] = output["topological_graph_matrix"][idx, :including_step_num, :including_step_num]

            # attention
            result["image_attention_weight"] = output["image_attention_weights"][idx, :, :actual_candidate_num]
            result["caption_attention_weight"] = output["caption_attention_weights"][idx, :, :actual_candidate_num]

            result["ground_truth_image_attention_weight"] = \
                output["scale_down_aggregate_attention_maps"][idx, :actual_candidate_num]
            result["ground_truth_caption_attention_weight"] = \
                output["aggregate_caption_attentions"][idx, :actual_candidate_num]

            # if opt.task_completion:
            #     result["img_completion_loss"] = output["loss"]["img_completion_loss"][idx]
            #     result["cap_completion_loss"] = output["loss"]["cap_completion_loss"][idx]

            results[data_i["information"][idx]["goal_index"]] = result

        # measure elapsed time
        batch_time.update(time.time() - end, n=1)
        end = time.time()

        if i % log_step == 0:

            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))

    return results

def eval(model_path, data_path=None, split='dev', save_path=None):
    """
    Evaluate a trained model on either dev or test.
    """
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
    else:
        gpu_num = 1
    assert gpu_num == 1 or gpu_num == 2, "Need to make sure the available gpu number is not larger than 2!"
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 4
    opt.batch_size = 8 * gpu_num
    opt.eval_name = os.path.join("/".join(model_path.split("/")[1:-1]), "eval")
    opt.eval_name = os.path.join("..", opt.eval_name)
    # opt.extend_graph = False
    if not hasattr(opt, "separate_attention"):
        opt.separate_attention = False

    logger.info(opt)

    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = BaselineManager(opt)

    # only eval in one GPU
    # Need to set CUDA_VISIBLE_DEVICES=0
    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    # build evaluator
    evaluator = Evaluator(opt, mode=split)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    logger.info('Loading dataset')
    data_loader = visualhow.get_test_loader(split, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)

    logger.info('Computing results...')
    with torch.no_grad():
        predictions = get_prediction(model, data_loader, opt, opt.log_step, logging.info)

    # show_heatmap(predictions[list(predictions.keys())[0]]["image_graph"][-1])
    start = time.time()

    evaluator.load_data(predictions)
    cur_metrics = evaluator.measure()

    end = time.time()

    # save predictions
    with open(os.path.join(opt.eval_name, 'prediction_{}.pkl'.format(split)), "wb") as f:
        pickle.dump(predictions, f)

    logger.info("calculate evaluation time: {}".format(end - start))

    logging.info("rsum: %.3f," % cur_metrics["rsum"])

    logging.info("-" * 20)
    logging.info("imrr: %.4f," % cur_metrics["imrr"])
    logging.info("ir1: %.3f," % cur_metrics["ir1"])
    logging.info("ir5: %.3f," % cur_metrics["ir5"])
    logging.info("ir10: %.3f," % cur_metrics["ir10"])
    logging.info("imeanr: %.3f," % cur_metrics["imeanr"])
    logging.info("imedr: %.3f," % cur_metrics["imedr"])

    logging.info("-" * 20)
    logging.info("cmrr: %.4f," % cur_metrics["cmrr"])
    logging.info("cr1: %.3f," % cur_metrics["cr1"])
    logging.info("cr5: %.3f," % cur_metrics["cr5"])
    logging.info("cr10: %.3f," % cur_metrics["cr10"])
    logging.info("cmeanr: %.3f," % cur_metrics["cmeanr"])
    logging.info("cmedr: %.3f," % cur_metrics["cmedr"])

    logging.info("-" * 20)
    logging.info("iauc: %.3f," % cur_metrics["iauc"])
    logging.info("iaupr: %.3f," % cur_metrics["iaupr"])
    logging.info("iiou025: %.3f," % cur_metrics["iiou025"])
    logging.info("iiou05: %.3f," % cur_metrics["iiou05"])
    logging.info("iiou075: %.3f," % cur_metrics["iiou075"])

    logging.info("-" * 20)
    logging.info("cauc: %.3f," % cur_metrics["cauc"])
    logging.info("caupr: %.3f," % cur_metrics["caupr"])
    logging.info("ciou025: %.3f," % cur_metrics["ciou025"])
    logging.info("ciou05: %.3f," % cur_metrics["ciou05"])
    logging.info("ciou075: %.3f," % cur_metrics["ciou075"])

    logging.info("-" * 20)
    logging.info("image_in_degree_completion: %.4f," % cur_metrics["image_in_degree_completion"])
    logging.info("image_out_degree_completion: %.4f," % cur_metrics["image_out_degree_completion"])
    logging.info("image_completion: %.4f," % cur_metrics["image_completion"])
    logging.info("caption_in_degree_completion: %.4f," % cur_metrics["caption_in_degree_completion"])
    logging.info("caption_out_degree_completion: %.4f," % cur_metrics["caption_out_degree_completion"])
    logging.info("caption_completion: %.4f," % cur_metrics["caption_completion"])

    logging.info("-" * 20)
    logging.info("iCC: %.3f," % cur_metrics["iCC"])
    logging.info("iSIM: %.3f," % cur_metrics["iSIM"])
    logging.info("iKLD: %.3f," % cur_metrics["iKLD"])
    logging.info("iSpearman: %.3f," % cur_metrics["iSpearman"])

    logging.info("-" * 20)
    logging.info("cCC: %.3f," % cur_metrics["cCC"])
    logging.info("cSIM: %.3f," % cur_metrics["cSIM"])
    logging.info("cKLD: %.3f," % cur_metrics["cKLD"])
    logging.info("cSpearman: %.3f," % cur_metrics["cSpearman"])

    with open(save_path, "w") as f:
        json.dump(cur_metrics, f, indent=2)

