import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lib.baseline import BaselineModel
from transformers import modeling_utils
from transformers import GPT2Tokenizer, ViltProcessor
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import scipy.stats

from torch.cuda.amp import GradScaler, autocast


epsilon = 1e-7

import logging

logger = logging.getLogger(__name__)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        # p.grad.data = p.grad.data.float()

class BaselineManager(nn.Module):
    # initializers
    def __init__(self, opt):
        super(BaselineManager, self).__init__()
        self.opt = opt

        self.baseline = BaselineModel(opt)

        # Creates a GradScaler once at the beginning of training.
        if self.opt.fp16:
            self.scaler = GradScaler()

        if torch.cuda.is_available():
            self.baseline.cuda()

        params = list(self.baseline.parameters())

        self.params = params

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.baseline.goal_enc.parameters()) + list(self.baseline.cap_enc.parameters())
                bert_params = list(self.baseline.goal_enc.bert.parameters()) + list(self.baseline.cap_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.baseline.img_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.baseline.graph_enc.parameters(), 'lr': opt.learning_rate},
                ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        else:
            if self.opt.optim == 'adam':
                all_text_params = list(self.baseline.goal_enc.parameters()) + list(self.baseline.cap_enc.parameters())
                bert_params = list(self.baseline.goal_enc.bert.parameters()) + list(self.baseline.cap_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.baseline.img_enc.backbone.top.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.baseline.img_enc.backbone.base.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.baseline.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    {'params': self.baseline.graph_enc.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD([
                    {'params': self.baseline.goal_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.baseline.cap_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.baseline.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.backbone_lr_factor,
                     'weight_decay': decay_factor},
                    {'params': self.baseline.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    {'params': self.baseline.graph_enc.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, momentum=0.9, nesterov=True)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        if self.opt.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.opt.num_epochs * self.opt.train_loader_size // 4)
        elif self.opt.lr_scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.opt.num_epochs * self.opt.train_loader_size)
        elif self.opt.lr_scheduler == 'seq':
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=0.1, total_iters=self.opt.warmup_epochs * self.opt.train_loader_size)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=(self.opt.num_epochs - self.opt.warmup_epochs) * self.opt.train_loader_size)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[scheduler1, scheduler2],
                milestones=[self.opt.warmup_epochs * self.opt.train_loader_size])
        else:
            raise ValueError('Invalid lr_scheduler option {}'.format(self.opt.lr_scheduler))


        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            modeling_utils.unwrap_model(self.baseline).baseline_loss.max_violation_on()
        else:
            modeling_utils.unwrap_model(self.baseline).baseline_loss.max_violation_off()

    def set_second_stage(self, include_second_stage):
        if include_second_stage:
            modeling_utils.unwrap_model(self.baseline).baseline_loss.include_second_stage_on()
        else:
            modeling_utils.unwrap_model(self.baseline).baseline_loss.include_second_stage_off()

    def train_start(self):
        """switch to train mode
        """
        self.baseline.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.baseline.eval()

    def make_data_parallel(self):
        self.baseline = nn.DataParallel(self.baseline)
        self.data_parallel = True
        logger.info('Model is data paralleled now.')

    def model_state_dict(self):
        state_dict = self.baseline.state_dict()
        return state_dict

    def optimizer_state_dict(self):
        state_dict = self.optimizer.state_dict()
        return state_dict

    def scheduler_state_dict(self):
        state_dict = self.scheduler.state_dict()
        return state_dict

    def load_model_state_dict(self, state_dict):
        self.baseline.load_state_dict(state_dict, strict=False)

    def load_optimizer_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def load_scheduler_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.baseline, nn.DataParallel):
                self.baseline.module.img_enc.freeze_backbone()
            else:
                self.baseline.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.baseline, nn.DataParallel):
                self.baseline.module.img_enc.unfreeze_backbone(fixed_blocks)
            else:
                self.baseline.img_enc.unfreeze_backbone(fixed_blocks)

    def inference(self, goal, caption_steps, images, aggregate_caption_attentions, scale_down_aggregate_attention_maps,
                    actual_problem_solving_step_indicator, topological_graph_matrix):
        output = self.baseline(goal, caption_steps, images, actual_problem_solving_step_indicator,
                               aggregate_caption_attentions, scale_down_aggregate_attention_maps, topological_graph_matrix)
        return output

    def train(self, goal, caption_steps, images, aggregate_caption_attentions, scale_down_aggregate_attention_maps,
                    actual_problem_solving_step_indicator, topological_graph_matrix, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # train
        self.optimizer.zero_grad()

        if self.opt.fp16:
            with autocast(dtype=torch.float16):
                output = self.baseline(goal, caption_steps, images, actual_problem_solving_step_indicator,
                                       aggregate_caption_attentions, scale_down_aggregate_attention_maps,
                                       topological_graph_matrix)

                img_loss = output["loss"]["img_loss"].sum()
                cap_loss = output["loss"]["cap_loss"].sum()
                loss = output["loss"]["loss"].sum()
                self.logger.update('img_loss', img_loss.data.item(), goal.size(0))
                self.logger.update('cap_loss', cap_loss.data.item(), goal.size(0))
                self.logger.update('loss', loss.data.item(), goal.size(0))

                if warmup_alpha is not None:
                    loss = loss * warmup_alpha

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            self.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()
            self.scheduler.step()

        else:
            output = self.baseline(goal, caption_steps, images, actual_problem_solving_step_indicator,
                                   aggregate_caption_attentions, scale_down_aggregate_attention_maps,
                                   topological_graph_matrix)

            img_topology_loss = output["loss"]["img_topology_loss"].sum()
            cap_topology_loss = output["loss"]["cap_topology_loss"].sum()
            self.logger.update('img_topology_loss', img_topology_loss.data.item(), goal.size(0))
            self.logger.update('cap_topology_loss', cap_topology_loss.data.item(), goal.size(0))
            if self.opt.contrastive_loss:
                img_contrastive_loss = output["loss"]["img_contrastive_loss"].sum()
                cap_contrastive_loss = output["loss"]["cap_contrastive_loss"].sum()
                self.logger.update('img_contrastive_loss', img_contrastive_loss.data.item(), goal.size(0))
                self.logger.update('cap_contrastive_loss', cap_contrastive_loss.data.item(), goal.size(0))
            if self.opt.task_completion and output["loss"].get("img_completion_loss", None) is not None:
                img_completion_loss = output["loss"]["img_completion_loss"].sum()
                cap_completion_loss = output["loss"]["cap_completion_loss"].sum()
                self.logger.update('img_completion_loss', img_completion_loss.data.item(), goal.size(0))
                self.logger.update('cap_completion_loss', cap_completion_loss.data.item(), goal.size(0))
            if self.opt.stepwise_task_completion and output["loss"].get("im_propagation_loss", None) is not None:
                im_propagation_loss = output["loss"]["im_propagation_loss"].sum()
                im_receive_loss = output["loss"]["im_receive_loss"].sum()
                cap_propagation_loss = output["loss"]["cap_propagation_loss"].sum()
                cap_receive_loss = output["loss"]["cap_receive_loss"].sum()
                self.logger.update('im_propagation_loss', im_propagation_loss.data.item(), goal.size(0))
                self.logger.update('im_receive_loss', im_receive_loss.data.item(), goal.size(0))
                self.logger.update('cap_propagation_loss', cap_propagation_loss.data.item(), goal.size(0))
                self.logger.update('cap_receive_loss', cap_receive_loss.data.item(), goal.size(0))
            if self.opt.attention_supervision:
                im_att_loss = output["loss"]["im_att_loss"].sum()
                cap_att_loss = output["loss"]["cap_att_loss"].sum()
                self.logger.update('im_att_loss', im_att_loss.data.item(), goal.size(0))
                self.logger.update('cap_att_loss', cap_att_loss.data.item(), goal.size(0))
            loss = output["loss"]["loss"].sum()
            self.logger.update('loss', loss.data.item(), goal.size(0))

            if warmup_alpha is not None:
                loss = loss * warmup_alpha

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()



    @property
    def is_data_parallel(self):
        return self.data_parallel
