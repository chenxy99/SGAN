"""Training script"""
import os
import time
import numpy as np
import torch
import json
from transformers import BertTokenizer
from transformers import modeling_utils

from lib.datasets import visualhow
from lib.manager import BaselineManager

import logging
from torch.utils.tensorboard import SummaryWriter
from lib.evaluation import AverageMeter, LogCollector, get_prediction
from lib.evaluator import Evaluator

import arguments
import pickle


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    if not os.path.exists(opt.eval_name):
        os.makedirs(opt.eval_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = SummaryWriter(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)


    train_loader, val_loader = visualhow.get_loaders(opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)
    opt.train_loader_size = len(train_loader)

    model = BaselineManager(opt)

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_metric']
            if not model.is_data_parallel:
                model.make_data_parallel()
            model.load_state_dict(checkpoint['model'])
            model.load_optimizer_state_dict(checkpoint['optimizer'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_metric {})"
                        .format(opt.resume, start_epoch, best_metric))
            # validate(opt, val_loader, model)
            if opt.reset_start_epoch:
                start_epoch = 0
            else:
                model.load_scheduler_state_dict(checkpoint['scheduler'])
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    if not model.is_data_parallel:
        model.make_data_parallel()

    # build evaluator
    evaluator = Evaluator(opt, mode='val')

    # Train the Model
    best_metric = 0
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        if opt.loss_type == "contrastive":
            if epoch >= opt.vse_mean_warmup_epochs:
                opt.max_violation = True
                model.set_max_violation(opt.max_violation)
            else:
                opt.max_violation = False
                model.set_max_violation(opt.max_violation)

        if opt.loss_type == "bce" or opt.loss_type == "ohem":
            if epoch >= opt.two_stage_warmup_epochs:
                opt.include_second_stage = True
                model.set_second_stage(opt.include_second_stage)
            else:
                opt.include_second_stage = False
                model.set_second_stage(opt.include_second_stage)

        # Set up the all warm-up options
        if opt.precomp_enc_type == 'backbone':
            if epoch < opt.embedding_warmup_epochs:
                model.freeze_backbone()
                logger.info('All backbone weights are frozen, only train the embedding layers')
            else:
                model.unfreeze_backbone(3)

            if epoch < opt.embedding_warmup_epochs:
                logger.info('Warm up the embedding layers')
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs:
                model.unfreeze_backbone(3)  # only train the last block of resnet backbone
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs * 2:
                model.unfreeze_backbone(2)
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs * 3:
                model.unfreeze_backbone(1)
            else:
                model.unfreeze_backbone(0)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader, tb_logger)

        # evaluate on validation set
        cur_metric = validate(opt, val_loader, model, evaluator, epoch, tb_logger)

        # remember best R@ sum and save checkpoint
        is_best = cur_metric > best_metric
        best_metric = max(cur_metric, best_metric)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'scheduler': model.scheduler.state_dict(),
            'best_metric': best_metric,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint.pth'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader, tb_logger):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    logger.info(
        'trainable parameters for image encoder: {}'.
            format(count_params(modeling_utils.unwrap_model(model.baseline).img_enc)))
    logger.info(
        'trainable parameters for goal encoder: {}'.
            format(count_params(modeling_utils.unwrap_model(model.baseline).goal_enc)))
    logger.info(
        'trainable parameters for caption encoder: {}'.
            format(count_params(modeling_utils.unwrap_model(model.baseline).cap_enc)))
    logger.info(
        'trainable parameters for graph encoder: {}'.
            format(count_params(modeling_utils.unwrap_model(model.baseline).graph_enc)))

    num_loader_iter = len(train_loader)

    end = time.time()
    # opt.viz = True
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end, n=1)

        # make sure train logger is used
        model.logger = train_logger

        # Get the data
        tmp = [train_data["goal"].to('cuda'), train_data["caption_steps"].to('cuda'),
               train_data["images"].to('cuda'), train_data["aggregate_caption_attentions"].to('cuda'),
               train_data["scale_down_aggregate_attention_maps"].to('cuda'),
               train_data["actual_problem_solving_step_indicator"].to('cuda'),
               train_data["topological_graph_matrix"].to('cuda')]

        goal, caption_steps, images, aggregate_caption_attentions, scale_down_aggregate_attention_maps,\
            actual_problem_solving_step_indicator, topological_graph_matrix = tmp

        if epoch < opt.warmup_epochs and "seq" != opt.lr_scheduler:
            warmup_alpha = (float(i + 1) + epoch * num_loader_iter) / (num_loader_iter * opt.warmup_epochs)
            model.train(goal, caption_steps, images, aggregate_caption_attentions, scale_down_aggregate_attention_maps,
                        actual_problem_solving_step_indicator, topological_graph_matrix, warmup_alpha=warmup_alpha)
        else:
            model.train(goal, caption_steps, images, aggregate_caption_attentions, scale_down_aggregate_attention_maps,
                        actual_problem_solving_step_indicator, topological_graph_matrix)

        # measure elapsed time
        batch_time.update(time.time() - end, n=1)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            if epoch < opt.warmup_epochs and "seq" != opt.lr_scheduler:
                logging.info('Current epoch-{}, the first epoch for training, warmup alpha {}'.format(epoch, warmup_alpha))
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.add_scalar('epoch', epoch, global_step=model.Eiters)
        tb_logger.add_scalar('step', i, global_step=model.Eiters)
        tb_logger.add_scalar('batch_time', batch_time.val,  global_step=model.Eiters)
        tb_logger.add_scalar('data_time', data_time.val,  global_step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)




def validate(opt, val_loader, model, evaluator, epoch, tb_logger):
    logger = logging.getLogger(__name__)
    # only use single GPU to run the validation
    model = modeling_utils.unwrap_model(model)
    model.val_start()
    with torch.no_grad():
        predictions = get_prediction(model, val_loader, opt, opt.log_step, logging.info)

    # save predictions
    with open(os.path.join(opt.eval_name, 'prediction_val.pkl'), "wb") as f:
        pickle.dump(predictions, f)

    start = time.time()

    evaluator.load_data(predictions)
    cur_metrics = evaluator.measure()

    end = time.time()
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

    logging.info("-" * 20)
    cur_metric = cur_metrics["rsum"]
    logger.info('Current metric [rsum] is {:.4f}'.format(cur_metric))

    # record metrics in tensorboard
    tb_logger.add_scalar('Evaluation/rsum', cur_metrics["rsum"], global_step=epoch)
    tb_logger.add_scalar('Evaluation/ir1', cur_metrics["ir1"], global_step=epoch)
    tb_logger.add_scalar('Evaluation/ir5', cur_metrics["ir5"], epoch)
    tb_logger.add_scalar('Evaluation/ir10', cur_metrics["ir10"], epoch)
    tb_logger.add_scalar('Evaluation/imedr', cur_metrics["imedr"], epoch)
    tb_logger.add_scalar('Evaluation/imeanr', cur_metrics["imeanr"], epoch)
    tb_logger.add_scalar('Evaluation/imrr', cur_metrics["imrr"], epoch)
    tb_logger.add_scalar('Evaluation/cr1', cur_metrics["cr1"], epoch)
    tb_logger.add_scalar('Evaluation/cr5', cur_metrics["cr5"], epoch)
    tb_logger.add_scalar('Evaluation/cr10', cur_metrics["cr10"], epoch)
    tb_logger.add_scalar('Evaluation/cmedr', cur_metrics["cmedr"], epoch)
    tb_logger.add_scalar('Evaluation/cmeanr', cur_metrics["cmeanr"], epoch)
    tb_logger.add_scalar('Evaluation/cmrr', cur_metrics["cmrr"], epoch)

    tb_logger.add_scalar('Evaluation/iauc', cur_metrics["iauc"], epoch)
    tb_logger.add_scalar('Evaluation/iaupr', cur_metrics["iaupr"], epoch)
    tb_logger.add_scalar('Evaluation/iiou025', cur_metrics["iiou025"], epoch)
    tb_logger.add_scalar('Evaluation/iiou05', cur_metrics["iiou05"], epoch)
    tb_logger.add_scalar('Evaluation/iiou075', cur_metrics["iiou075"], epoch)
    tb_logger.add_scalar('Evaluation/cauc', cur_metrics["cauc"], epoch)
    tb_logger.add_scalar('Evaluation/caupr', cur_metrics["caupr"], epoch)
    tb_logger.add_scalar('Evaluation/ciou025', cur_metrics["ciou025"], epoch)
    tb_logger.add_scalar('Evaluation/ciou05', cur_metrics["ciou05"], epoch)
    tb_logger.add_scalar('Evaluation/ciou075', cur_metrics["ciou075"], epoch)

    tb_logger.add_scalar('Evaluation/image_in_degree_completion', cur_metrics["image_in_degree_completion"], epoch)
    tb_logger.add_scalar('Evaluation/image_out_degree_completion', cur_metrics["image_out_degree_completion"], epoch)
    tb_logger.add_scalar('Evaluation/image_completion', cur_metrics["image_completion"], epoch)
    tb_logger.add_scalar('Evaluation/caption_in_degree_completion', cur_metrics["caption_in_degree_completion"], epoch)
    tb_logger.add_scalar('Evaluation/caption_out_degree_completion', cur_metrics["caption_out_degree_completion"], epoch)
    tb_logger.add_scalar('Evaluation/caption_completion', cur_metrics["caption_completion"], epoch)

    tb_logger.add_scalar('Evaluation/image_CC', cur_metrics["iCC"], epoch)
    tb_logger.add_scalar('Evaluation/image_SIM', cur_metrics["iSIM"], epoch)
    tb_logger.add_scalar('Evaluation/image_KLD', cur_metrics["iKLD"], epoch)
    tb_logger.add_scalar('Evaluation/image_Spearman', cur_metrics["iSpearman"], epoch)

    tb_logger.add_scalar('Evaluation/caption_CC', cur_metrics["cCC"], epoch)
    tb_logger.add_scalar('Evaluation/caption_SIM', cur_metrics["cSIM"], epoch)
    tb_logger.add_scalar('Evaluation/caption_KLD', cur_metrics["cKLD"], epoch)
    tb_logger.add_scalar('Evaluation/caption_Spearman', cur_metrics["cSpearman"], epoch)


    return cur_metric


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()