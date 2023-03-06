import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import numpy as np
import math
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def cal_sim_score(pred, gt, mask=None):
    eps = 1e-15
    if mask is not None:
        gt[mask > 0] = gt[mask > 0] + eps
    else:
        gt = gt + eps
    gt = gt / gt.sum(-1, keepdim=True)

    sim_score = torch.minimum(pred, gt).sum(-1)

    return sim_score


# JSD Jensen–Shannon divergence score
def cal_jsd_score(pred, gt, mask=None):
    eps = 1e-15
    batch = len(pred)
    if mask is not None:
        gt[mask > 0] = gt[mask > 0] + eps
    else:
        gt = gt + eps
    gt = gt / gt.sum(-1, keepdim=True)
    m = 1 / 2 * (pred + gt)
    jsd_loss = 1 / 2 * (pred * torch.log(pred / (m + eps) + eps) + gt * torch.log(gt / (m + eps) + eps))
    if mask is None:
        loss = jsd_loss.sum(-1)
    else:
        loss = (jsd_loss * mask).sum(-1)
    score = 1 - loss / math.log(2)
    return score

# attention CE
def attention_celoss(pred, gt, mask=None):
    eps = 1e-15
    batch = len(pred)
    pred = pred.view(batch,-1)
    gt = gt.view(batch,-1)
    if mask is not None:
        gt[mask > 0] = gt[mask > 0] + eps
    else:
        gt = gt + eps
    gt = gt / gt.sum(-1, keepdim=True)
    if mask is not None:
        mask = mask.view(batch,-1)
        pred = pred[mask == 1]
        gt = gt[mask == 1]
    loss = -(gt*torch.log(torch.clamp(pred,min=eps,max=1))).sum(-1)
    return loss.sum() / batch

# attention KLD
def attention_kldloss(pred, gt, mask=None):
    eps = 1e-15
    batch = len(pred)
    pred = pred.view(batch, -1)
    gt = gt.view(batch, -1)
    if mask is not None:
        gt[mask > 0] = gt[mask > 0] + eps
    else:
        gt = gt + eps
    gt = gt / gt.sum(-1, keepdim=True)
    if mask is not None:
        mask = mask.view(batch,-1)
        pred = pred[mask == 1]
        gt = gt[mask == 1]
    loss = torch.mul(gt,torch.log(gt / (pred + eps) + eps)).sum(-1)
    return loss.sum() / batch

# attention balanced BCE
def attention_bceloss(pred, gt, mask=None):
    eps = 1e-15
    batch = len(pred)
    pred = pred.view(batch,-1)
    gt = gt.view(batch,-1)
    if mask is not None:
        mask = mask.view(batch,-1)
        pred = pred[mask == 1]
        gt = gt[mask == 1]
    neg_weight = gt.mean(-1,keepdim=True)
    pos_weight = 1 - neg_weight
    loss = -(pos_weight*gt*torch.log(torch.clamp(pred,min=eps,max=1))+neg_weight*(1-gt)*torch.log(torch.clamp(1-pred,min=eps,max=1))).mean(-1)
    return loss

class AttentionLoss(nn.Module):
    """
    Compute attention loss (binary cross entropy loss)
    """

    def __init__(self, opt):
        super(AttentionLoss, self).__init__()
        self.opt = opt
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor
        # self.attention_loss_type = self.opt.attention_loss_type


    def forward(self, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask):
        im_att_selection = []
        cap_att_selection = []

        for idx in range(gt_dep.shape[0]):
            cur_step_index = gt_dep[idx, 0, 2:]
            im_att_selection.append(im_att[idx, :, cur_step_index == 1])
            cap_att_selection.append(cap_att[idx, :, cur_step_index == 1])
        im_att_selection = torch.cat(im_att_selection, dim=1)
        cap_att_selection = torch.cat(cap_att_selection, dim=1)

        if self.progressive_optimization:
            layer_num = im_att.shape[1]
            img_attention_loss_list = []
            cap_attention_loss_list = []

            for idx in range(layer_num):
                img_attention_loss_list.append(attention_kldloss(im_att_selection[idx], im_gt_att))
                cap_attention_loss_list.append(attention_kldloss(cap_att_selection[idx], cap_gt_att, cap_emb_mask))

            im_att_loss = 0
            cap_att_loss = 0
            for idx in range(layer_num):
                im_att_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_attention_loss_list[idx]
                cap_att_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_attention_loss_list[idx]

        else:
            im_att_selection = im_att_selection[-1]
            cap_att_selection = cap_att_selection[-1]

            im_att_loss = attention_kldloss(im_att_selection, im_gt_att)
            cap_att_loss = attention_kldloss(cap_att_selection, cap_gt_att, cap_emb_mask)


        return im_att_loss, cap_att_loss


class BaselineLoss(nn.Module):
    """
    Compute baseline loss (binary cross entropy loss)
    """

    def __init__(self, opt):
        super(BaselineLoss, self).__init__()
        self.opt = opt
        if self.opt.fp16:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion = nn.BCELoss(reduction="none")
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor

        self.task_completion = self.opt.task_completion
        self.task_completion_weight = self.opt.task_completion_weight
        if self.task_completion:
            self.task_completion_loss = TaskCompletionLoss(opt)
        self.attention_supervision = self.opt.attention_supervision
        self.attention_supervision_weight = self.opt.attention_supervision_weight
        if self.attention_supervision:
            self.attention_loss = AttentionLoss(opt)
        self.balance_loss = opt.balance_loss

    def compute_loss(self, pred, gt, gt_dep_mask=None):
        if self.balance_loss == False:
            if self.opt.fp16:
                eps = 1e-16
                pred = torch.log((pred + eps) / (1 - pred + eps))
                if gt_dep_mask is not None:
                    pred = pred[gt_dep_mask==1]
                    gt = gt[gt_dep_mask==1]
                sample_loss = self.criterion(pred, gt)
            else:
                if gt_dep_mask is not None:
                    pred = pred[gt_dep_mask==1]
                    gt = gt[gt_dep_mask==1]
                sample_loss = self.criterion(pred, gt)

            loss = sample_loss.mean()
        else:
            if self.opt.fp16:
                eps = 1e-16
                pred = torch.log((pred + eps) / (1 - pred + eps))
                if gt_dep_mask is not None:
                    pred = pred[gt_dep_mask==1]
                    gt = gt[gt_dep_mask==1]
                sample_loss = self.criterion(pred, gt)
            else:
                if gt_dep_mask is not None:
                    pred = pred[gt_dep_mask==1]
                    gt = gt[gt_dep_mask==1]
                sample_loss = self.criterion(pred, gt)
            neg_weight = gt.mean(-1, keepdim=True).mean(-2, keepdim=True)
            pos_weight = 1 - neg_weight

            loss = (pos_weight * gt * sample_loss + neg_weight * (1 - gt) * sample_loss).mean()

        return loss


    def forward(self, im_dep, cap_dep, cap_emb_mask, gt_dep, im_att=None, cap_att=None, im_gt_att=None, cap_gt_att=None, gt_dep_mask=None):
        # calculate the topology loss
        if self.progressive_optimization:
            layer_num = im_dep.shape[1]
            img_topology_loss_list = []
            cap_topology_loss_list = []

            for idx in range(layer_num):
                img_topology_loss_list.append(self.compute_loss(im_dep[:, idx], gt_dep.float(), gt_dep_mask))
                cap_topology_loss_list.append(self.compute_loss(cap_dep[:, idx], gt_dep.float(), gt_dep_mask))

            img_topology_loss = 0
            cap_topology_loss = 0
            for idx in range(layer_num):
                img_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_topology_loss_list[idx]
                cap_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_topology_loss_list[idx]

        else:
            im_dep = im_dep[:, -1]
            cap_dep = cap_dep[:, -1]

            img_topology_loss = self.compute_loss(im_dep, gt_dep.float(), gt_dep_mask)
            cap_topology_loss = self.compute_loss(cap_dep, gt_dep.float(), gt_dep_mask)

        topology_loss = img_topology_loss + cap_topology_loss

        loss = topology_loss

        data = {
            "img_topology_loss": img_topology_loss,
            "cap_topology_loss": cap_topology_loss,
        }

        # calculate the task completion loss
        if self.task_completion:
            img_completion_loss, cap_completion_loss = \
                self.task_completion_loss(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            completion_loss = img_completion_loss + cap_completion_loss

            loss += self.opt.task_completion_weight * completion_loss

            data["img_completion_loss"] = img_completion_loss
            data["cap_completion_loss"] = cap_completion_loss

        if self.attention_supervision:
            im_att_loss, cap_att_loss = \
                self.attention_loss(gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            att_loss = im_att_loss + cap_att_loss

            loss += self.opt.attention_supervision_weight * att_loss

            data["im_att_loss"] = im_att_loss
            data["cap_att_loss"] = cap_att_loss

        data["loss"] = loss

        return data


class BaselineTwoStageLoss(nn.Module):
    """
    Compute baseline loss (binary cross entropy loss)
    """

    def __init__(self, opt):
        super(BaselineTwoStageLoss, self).__init__()
        self.opt = opt
        if self.opt.fp16:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion = nn.BCELoss(reduction="none")
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor

        self.task_completion = self.opt.task_completion
        self.task_completion_weight = self.opt.task_completion_weight
        if self.task_completion:
            self.task_completion_loss = TaskCompletionLoss(opt)
        self.attention_supervision = self.opt.attention_supervision
        self.attention_supervision_weight = self.opt.attention_supervision_weight
        if self.attention_supervision:
            self.attention_loss = AttentionLoss(opt)
        self.balance_loss = self.opt.balance_loss
        self.include_second_stage = False

    def include_second_stage_on(self):
        self.include_second_stage = True
        print('Use two stage objective.')

    def include_second_stage_off(self):
        self.include_second_stage = False
        print('Use one stage objective.')

    def compute_loss(self, pred, gt):
        if self.opt.fp16:
            eps = 1e-16
            pred = torch.log((pred + eps) / (1 - pred + eps))
            # for the retrieval task
            retrieval_loss = self.criterion(pred[:, 0], gt[:, 0])
            # for the dependency relationship
            dependency_loss = self.criterion(pred, gt)
        else:
            # for the retrieval task
            retrieval_loss = self.criterion(pred[:, 0], gt[:, 0])
            # for the dependency relationship
            dependency_loss = self.criterion(pred, gt)

        if self.balance_loss:
            retrieval_neg_weight = gt[:, 0].mean(-1, keepdim=True)
            retrieval_pos_weight = 1 - retrieval_neg_weight

            dependency_neg_weight = gt.mean(-1, keepdim=True).mean(-2, keepdim=True)
            dependency_pos_weight = 1 - dependency_neg_weight

            retrieval_loss = (retrieval_pos_weight * gt[:, 0] * retrieval_loss + retrieval_neg_weight * (1 - gt[:, 0]) * retrieval_loss)
            dependency_loss = (dependency_pos_weight * gt * dependency_loss + dependency_neg_weight * (1 - gt) * dependency_loss)

        if self.include_second_stage:
            loss = dependency_loss.mean()
        else:
            loss = retrieval_loss.mean()

        return loss


    def forward(self, im_dep, cap_dep, cap_emb_mask, gt_dep, im_att=None, cap_att=None, im_gt_att=None, cap_gt_att=None):
        # calculate the topology loss
        if self.progressive_optimization:
            layer_num = im_dep.shape[1]
            img_topology_loss_list = []
            cap_topology_loss_list = []

            for idx in range(layer_num):
                img_topology_loss_list.append(self.compute_loss(im_dep[:, idx], gt_dep.float()))
                cap_topology_loss_list.append(self.compute_loss(cap_dep[:, idx], gt_dep.float()))

            img_topology_loss = 0
            cap_topology_loss = 0
            for idx in range(layer_num):
                img_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_topology_loss_list[idx]
                cap_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_topology_loss_list[idx]

        else:
            im_dep = im_dep[:, -1]
            cap_dep = cap_dep[:, -1]

            img_topology_loss = self.compute_loss(im_dep, gt_dep.float())
            cap_topology_loss = self.compute_loss(cap_dep, gt_dep.float())

        topology_loss = img_topology_loss + cap_topology_loss

        loss = topology_loss

        data = {
            "img_topology_loss": img_topology_loss,
            "cap_topology_loss": cap_topology_loss,
        }

        # calculate the task completion loss
        if self.task_completion:
            img_completion_loss, cap_completion_loss = \
                self.task_completion_loss(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            completion_loss = img_completion_loss + cap_completion_loss

            loss += self.opt.task_completion_weight * completion_loss

            data["img_completion_loss"] = img_completion_loss
            data["cap_completion_loss"] = cap_completion_loss

        if self.attention_supervision:
            im_att_loss, cap_att_loss = \
                self.attention_loss(gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            att_loss = im_att_loss + cap_att_loss

            loss += self.opt.attention_supervision_weight * att_loss

            data["im_att_loss"] = im_att_loss
            data["cap_att_loss"] = cap_att_loss

        data["loss"] = loss

        return data


class RankingLoss(nn.Module):
    """
    Compute ranking loss (max-margin based)
    """

    def __init__(self, opt):
        super(RankingLoss, self).__init__()
        self.opt = opt
        self.margin = self.opt.margin
        if self.opt.fp16:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion = nn.BCELoss(reduction="none")
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor

        self.task_completion = self.opt.task_completion
        self.task_completion_weight = self.opt.task_completion_weight
        if self.task_completion:
            self.task_completion_loss = TaskCompletionLoss(opt)
        self.attention_supervision = self.opt.attention_supervision
        self.attention_supervision_weight = self.opt.attention_supervision_weight
        if self.attention_supervision:
            self.attention_loss = AttentionLoss(opt)
        self.max_violation = False

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def compute_loss(self, pred, gt, gt_dep_mask=None):
        # for the retrieval task
        retrieval_pred = pred[:, 0, 2:]
        retrieval_gt = gt[:, 0, 2:]

        diagonal = retrieval_pred[retrieval_gt == 1]
        # compare every diagonal score to scores in its row
        # task retrieval
        cost_task = (self.margin + retrieval_pred - diagonal.unsqueeze(0)).clamp(min=0)
        # clear diagonals
        mask = retrieval_gt > .5
        cost_task = cost_task.masked_fill_(mask, 0)

        # compare every diagonal score to scores in its column
        # steps retrieval
        extend_retrieval_pred = []
        extend_retrieval_gt = []
        for idx in range(retrieval_gt.shape[0]):
            extend_retrieval_pred.append(
                retrieval_pred[idx].unsqueeze(0).repeat(int(retrieval_gt[idx].sum().item()), 1))
            extend_retrieval_gt.append(
                retrieval_gt[idx].unsqueeze(0).repeat(int(retrieval_gt[idx].sum().item()), 1))
        extend_retrieval_pred = torch.cat(extend_retrieval_pred, dim=0)
        extend_retrieval_gt = torch.cat(extend_retrieval_gt, dim=0)
        cost_step = (self.margin + extend_retrieval_pred - diagonal.unsqueeze(1)).clamp(min=0)
        # clear diagonals
        mask = extend_retrieval_gt > .5
        cost_step = cost_step.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_task = cost_task.max(0)[0]
            cost_step = cost_step.max(1)[0]

        # for the dependency relationship
        extended_gt = gt.clone()
        extended_gt[:, 0, 1] = 1

        out_degree_pred = []
        out_degree_gt = []
        for idx in range(retrieval_gt.shape[0]):
            out_degree_pred.append(pred[idx, extended_gt[idx, 0] == 1])
            out_degree_gt.append(gt[idx, extended_gt[idx, 0] == 1])
        out_degree_pred = torch.cat(out_degree_pred, 0)
        out_degree_gt = torch.cat(out_degree_gt, 0)

        # extend the matrix
        extended_out_degree_pred = []
        extended_out_degree_gt = []
        for idx in range(out_degree_gt.shape[0]):
            extended_out_degree_pred.\
                append(out_degree_pred[idx].unsqueeze(0).repeat(int(out_degree_gt[idx].sum().item()), 1))
            extended_out_degree_gt. \
                append(out_degree_gt[idx].unsqueeze(0).repeat(int(out_degree_gt[idx].sum().item()), 1))
        extended_out_degree_pred =  torch.cat(extended_out_degree_pred, 0)
        extended_out_degree_gt = torch.cat(extended_out_degree_gt, 0)

        diagonal = out_degree_pred[out_degree_gt == 1]
        # compare every diagonal score to scores in its row
        # out degree retrieval
        cost_out_degree = (self.margin + extended_out_degree_pred - diagonal.unsqueeze(1)).clamp(min=0)
        # clear diagonals
        mask = extended_out_degree_gt > .5
        cost_out_degree = cost_out_degree.masked_fill_(mask, 0)

        in_degree_pred = []
        in_degree_gt = []
        for idx in range(retrieval_gt.shape[0]):
            in_degree_pred.append(pred[idx, :, extended_gt[idx, 0] == 1])
            in_degree_gt.append(gt[idx, :, extended_gt[idx, 0] == 1])
        in_degree_pred = torch.cat(in_degree_pred, 1)
        in_degree_gt = torch.cat(in_degree_gt, 1)
        in_degree_pred = in_degree_pred.t().contiguous()
        in_degree_gt = in_degree_gt.t().contiguous()
        # extend the matrix
        extended_in_degree_pred = []
        extended_in_degree_gt = []
        for idx in range(in_degree_gt.shape[0]):
            extended_in_degree_pred. \
                append(in_degree_pred[idx].unsqueeze(0).repeat(int(in_degree_gt[idx].sum().item()), 1))
            extended_in_degree_gt. \
                append(in_degree_gt[idx].unsqueeze(0).repeat(int(in_degree_gt[idx].sum().item()), 1))
        extended_in_degree_pred = torch.cat(extended_in_degree_pred, 0)
        extended_in_degree_gt = torch.cat(extended_in_degree_gt, 0)

        diagonal = in_degree_pred[in_degree_gt == 1]
        # compare every diagonal score to scores in its row
        # in degree retrieval
        cost_in_degree = (self.margin + extended_in_degree_pred - diagonal.unsqueeze(1)).clamp(min=0)
        # clear diagonals
        mask = extended_in_degree_gt > .5
        cost_in_degree = cost_in_degree.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_out_degree = cost_out_degree.max(1)[0]
            cost_in_degree = cost_in_degree.max(1)[0]

        cost_task = cost_task.view(-1)
        cost_step = cost_step.view(-1)
        cost_out_degree = cost_out_degree.view(-1)
        cost_in_degree = cost_in_degree.view(-1)
        loss = (cost_task.sum() + cost_step.sum() + cost_out_degree.sum() + cost_in_degree.sum()) \
               / (cost_task.shape[0] + cost_step.shape[0] + cost_out_degree.shape[0] + cost_in_degree.shape[0])

        return loss


    def forward(self, im_dep, cap_dep, cap_emb_mask, gt_dep, im_att=None, cap_att=None, im_gt_att=None, cap_gt_att=None, gt_dep_mask=None):
        # calculate the topology loss
        if self.progressive_optimization:
            layer_num = im_dep.shape[1]
            img_topology_loss_list = []
            cap_topology_loss_list = []

            for idx in range(layer_num):
                img_topology_loss_list.append(self.compute_loss(im_dep[:, idx], gt_dep.float(), gt_dep_mask))
                cap_topology_loss_list.append(self.compute_loss(cap_dep[:, idx], gt_dep.float(), gt_dep_mask))

            img_topology_loss = 0
            cap_topology_loss = 0
            for idx in range(layer_num):
                img_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_topology_loss_list[idx]
                cap_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_topology_loss_list[idx]

        else:
            im_dep = im_dep[:, -1]
            cap_dep = cap_dep[:, -1]

            img_topology_loss = self.compute_loss(im_dep, gt_dep.float(), gt_dep_mask)
            cap_topology_loss = self.compute_loss(cap_dep, gt_dep.float(), gt_dep_mask)

        topology_loss = img_topology_loss + cap_topology_loss

        loss = topology_loss

        data = {
            "img_topology_loss": img_topology_loss,
            "cap_topology_loss": cap_topology_loss,
        }

        # calculate the task completion loss
        if self.task_completion:
            img_completion_loss, cap_completion_loss = \
                self.task_completion_loss(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            completion_loss = img_completion_loss + cap_completion_loss

            loss += self.opt.task_completion_weight * completion_loss

            data["img_completion_loss"] = img_completion_loss
            data["cap_completion_loss"] = cap_completion_loss

        if self.attention_supervision:
            im_att_loss, cap_att_loss = \
                self.attention_loss(gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            att_loss = im_att_loss + cap_att_loss

            loss += self.opt.attention_supervision_weight * att_loss

            data["im_att_loss"] = im_att_loss
            data["cap_att_loss"] = cap_att_loss

        data["loss"] = loss

        return data


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor

    def forward(self, im_dep, cap_dep, gt_dep):
        eps = 1e-15
        contrastive_selection = []
        for idx in range(gt_dep.shape[0]):
            cur_step_index = dense_to_sparse(gt_dep[idx])[0]
            # remove start point
            cur_step_index = cur_step_index[:, cur_step_index[0] != 0]
            # remove end point
            cur_step_index = cur_step_index[:, cur_step_index[1] != 1]
            contrastive_selection.append(cur_step_index)

        if self.progressive_optimization:
            layer_num = im_dep.shape[1]
            img_contrastive_loss_list = []
            cap_contrastive_loss_list = []

            for idx in range(layer_num):
                im_dep_selection = im_dep[:, idx]
                cap_dep_selection = cap_dep[:, idx]

                im_contrastive_loss = []
                cap_contrastive_loss = []
                for idx in range(gt_dep.shape[0]):
                    pos_selection = contrastive_selection[idx]
                    # image
                    pos_prob = im_dep_selection[idx, pos_selection[0], pos_selection[1]]
                    neg_prob = im_dep_selection[idx, pos_selection[1], pos_selection[0]]
                    contrastive_loss = -torch.log(pos_prob / (pos_prob + neg_prob + eps) + eps)
                    im_contrastive_loss.append(contrastive_loss)
                    # caption
                    pos_prob = cap_dep_selection[idx, pos_selection[0], pos_selection[1]]
                    neg_prob = cap_dep_selection[idx, pos_selection[1], pos_selection[0]]
                    contrastive_loss = -torch.log(pos_prob / (pos_prob + neg_prob + eps) + eps)
                    cap_contrastive_loss.append(contrastive_loss)
                im_contrastive_loss = torch.cat(im_contrastive_loss, dim=0)
                cap_contrastive_loss = torch.cat(cap_contrastive_loss, dim=0)
                if im_contrastive_loss.shape[0] == 0:
                    im_contrastive_loss = gt_dep.new_zeros(1)[0]
                    cap_contrastive_loss = gt_dep.new_zeros(1)[0]
                else:
                    im_contrastive_loss = im_contrastive_loss.mean()
                    cap_contrastive_loss = cap_contrastive_loss.mean()
                img_contrastive_loss_list.append(im_contrastive_loss)
                cap_contrastive_loss_list.append(cap_contrastive_loss)

            im_contrastive_loss = 0
            cap_contrastive_loss = 0
            for idx in range(layer_num):
                im_contrastive_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_contrastive_loss_list[idx]
                cap_contrastive_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_contrastive_loss_list[idx]

        else:
            im_dep_selection = im_dep[:, -1]
            cap_dep_selection = cap_dep[:, -1]

            im_contrastive_loss = []
            cap_contrastive_loss = []
            for idx in range(gt_dep.shape[0]):
                pos_selection = contrastive_selection[idx]
                # image
                pos_prob = im_dep_selection[idx, pos_selection[0], pos_selection[1]]
                neg_prob = im_dep_selection[idx, pos_selection[1], pos_selection[0]]
                contrastive_loss = -torch.log(pos_prob / (pos_prob + neg_prob + eps) + eps)
                im_contrastive_loss.append(contrastive_loss)
                # caption
                pos_prob = cap_dep_selection[idx, pos_selection[0], pos_selection[1]]
                neg_prob = cap_dep_selection[idx, pos_selection[1], pos_selection[0]]
                contrastive_loss = -torch.log(pos_prob / (pos_prob + neg_prob + eps) + eps)
                cap_contrastive_loss.append(contrastive_loss)
            im_contrastive_loss = torch.cat(im_contrastive_loss, dim=0)
            cap_contrastive_loss = torch.cat(cap_contrastive_loss, dim=0)
            if im_contrastive_loss.shape[0] == 0:
                im_contrastive_loss = gt_dep.new_zeros(1)[0]
                cap_contrastive_loss = gt_dep.new_zeros(1)[0]
            else:
                im_contrastive_loss = im_contrastive_loss.mean()
                cap_contrastive_loss = cap_contrastive_loss.mean()

        return im_contrastive_loss, cap_contrastive_loss


class FocalLoss(nn.Module):
    """
    Compute focal loss (binary cross entropy loss)
    """

    def __init__(self, opt, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.opt = opt
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = torchvision.ops.sigmoid_focal_loss

        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor

        self.task_completion = self.opt.task_completion
        self.task_completion_weight = self.opt.task_completion_weight
        if self.task_completion:
            self.task_completion_loss = TaskCompletionLoss(opt)
        self.attention_supervision = self.opt.attention_supervision
        self.attention_supervision_weight = self.opt.attention_supervision_weight
        if self.attention_supervision:
            self.attention_loss = AttentionLoss(opt)

    def forward(self, im_dep, cap_dep, cap_emb_mask, gt_dep, im_att=None, cap_att=None, im_gt_att=None, cap_gt_att=None):
        # calculate the topology loss
        if self.progressive_optimization:
            layer_num = im_dep.shape[1]
            img_topology_loss_list = []
            cap_topology_loss_list = []

            for idx in range(layer_num):
                img_topology_loss_list.append(self.compute_loss(im_dep[:, idx], gt_dep.float()))
                cap_topology_loss_list.append(self.compute_loss(cap_dep[:, idx], gt_dep.float()))

            img_topology_loss = 0
            cap_topology_loss = 0
            for idx in range(layer_num):
                img_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_topology_loss_list[idx]
                cap_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_topology_loss_list[idx]

        else:
            im_dep = im_dep[:, -1]
            cap_dep = cap_dep[:, -1]

            img_topology_loss = self.compute_loss(im_dep, gt_dep.float())
            cap_topology_loss = self.compute_loss(cap_dep, gt_dep.float())

        topology_loss = img_topology_loss + cap_topology_loss

        loss = topology_loss

        data = {
            "img_topology_loss": img_topology_loss,
            "cap_topology_loss": cap_topology_loss,
        }

        # calculate the task completion loss
        if self.task_completion:
            img_completion_loss, cap_completion_loss = \
                self.task_completion_loss(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            completion_loss = img_completion_loss + cap_completion_loss

            loss += self.opt.task_completion_weight * completion_loss

            data["img_completion_loss"] = img_completion_loss
            data["cap_completion_loss"] = cap_completion_loss

        if self.attention_supervision:
            im_att_loss, cap_att_loss = \
                self.attention_loss(gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            att_loss = im_att_loss + cap_att_loss

            loss += self.opt.attention_supervision_weight * att_loss

            data["im_att_loss"] = im_att_loss
            data["cap_att_loss"] = cap_att_loss

        data["loss"] = loss

        return data

    def compute_loss(self, pred, gt):
        loss = self.criterion(pred, gt.float(), reduction="mean")
        return loss

class OHEMLoss(nn.Module):
    """
    Compute Online hard sample mining loss (binary cross entropy loss)
    """

    def __init__(self, opt, ratio=3):
        super(OHEMLoss, self).__init__()
        self.opt = opt
        self.ratio = ratio
        if self.opt.fp16:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion = nn.BCELoss(reduction="none")
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor

        self.task_completion = self.opt.task_completion
        self.task_completion_weight = self.opt.task_completion_weight
        if self.task_completion:
            self.task_completion_loss = TaskCompletionLoss(opt)
        self.attention_supervision = self.opt.attention_supervision
        self.attention_supervision_weight = self.opt.attention_supervision_weight
        if self.attention_supervision:
            self.attention_loss = AttentionLoss(opt)


    def forward(self, im_dep, cap_dep, cap_emb_mask, gt_dep, im_att=None, cap_att=None, im_gt_att=None, cap_gt_att=None):
        # calculate the topology loss
        if self.progressive_optimization:
            layer_num = im_dep.shape[1]
            img_topology_loss_list = []
            cap_topology_loss_list = []

            for idx in range(layer_num):
                img_topology_loss_list.append(self.compute_loss(im_dep[:, idx], gt_dep.float()))
                cap_topology_loss_list.append(self.compute_loss(cap_dep[:, idx], gt_dep.float()))

            img_topology_loss = 0
            cap_topology_loss = 0
            for idx in range(layer_num):
                img_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_topology_loss_list[idx]
                cap_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_topology_loss_list[idx]

        else:
            im_dep = im_dep[:, -1]
            cap_dep = cap_dep[:, -1]

            img_topology_loss = self.compute_loss(im_dep, gt_dep.float())
            cap_topology_loss = self.compute_loss(cap_dep, gt_dep.float())

        topology_loss = img_topology_loss + cap_topology_loss

        loss = topology_loss

        data = {
            "img_topology_loss": img_topology_loss,
            "cap_topology_loss": cap_topology_loss,
        }

        # calculate the task completion loss
        if self.task_completion:
            img_completion_loss, cap_completion_loss = \
                self.task_completion_loss(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            completion_loss = img_completion_loss + cap_completion_loss

            loss += self.opt.task_completion_weight * completion_loss

            data["img_completion_loss"] = img_completion_loss
            data["cap_completion_loss"] = cap_completion_loss

        if self.attention_supervision:
            im_att_loss, cap_att_loss = \
                self.attention_loss(gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            att_loss = im_att_loss + cap_att_loss

            loss += self.opt.attention_supervision_weight * att_loss

            data["im_att_loss"] = im_att_loss
            data["cap_att_loss"] = cap_att_loss

        data["loss"] = loss

        return data

    def compute_loss(self, pred, gt):
        loss = []
        if self.opt.fp16:
            eps = 1e-16
            pred = torch.log((pred + eps) / (1 - pred + eps))
            sample_loss = self.criterion(pred, gt)
        else:
            sample_loss = self.criterion(pred, gt)
        pos_sample = (gt.sum(-1).sum(-1)).int()
        neg_sample = pos_sample * self.ratio
        for idx in range(len(pos_sample)):
            pos_loss = sample_loss[idx][gt[idx] == 1]
            neg_loss = sample_loss[idx][gt[idx] == 0]
            neg_loss = neg_loss.topk(neg_sample[idx])[0]
            loss.append(pos_loss)
            loss.append(neg_loss)

        loss = torch.cat(loss, dim=0)
        loss = loss.mean()

        return loss


class OHEMTwoStageLoss(nn.Module):
    """
    Compute Online hard sample mining loss (binary cross entropy loss)
    """

    def __init__(self, opt, ratio=3):
        super(OHEMTwoStageLoss, self).__init__()
        self.opt = opt
        self.ratio = ratio
        if self.opt.fp16:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion = nn.BCELoss(reduction="none")
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor

        self.task_completion = self.opt.task_completion
        self.stepwise_task_completion = self.opt.stepwise_task_completion
        self.task_completion_weight = self.opt.task_completion_weight
        if self.task_completion:
            self.task_completion_loss = TaskCompletionLoss(opt)
        self.stepwise_task_completion_weight = self.opt.stepwise_task_completion_weight
        if self.stepwise_task_completion:
            self.stepwise_task_completion_loss = StepwiseTaskCompletionLoss(opt)
        self.attention_supervision = self.opt.attention_supervision
        self.attention_supervision_weight = self.opt.attention_supervision_weight
        if self.attention_supervision:
            self.attention_loss = AttentionLoss(opt)
        self.include_second_stage = False
        self.contrastive_loss = self.opt.contrastive_loss
        self.contrastive_loss_weight = self.opt.contrastive_loss_weight
        if self.contrastive_loss:
            self.contrastive_supervision_loss = ContrastiveLoss(opt)
        self.self_dependency = self.opt.self_dependency

    def include_second_stage_on(self):
        self.include_second_stage = True
        print('Use two stage objective.')

    def include_second_stage_off(self):
        self.include_second_stage = False
        print('Use one stage objective.')

    def forward(self, im_dep, cap_dep, cap_emb_mask, gt_dep, im_att=None, cap_att=None, im_gt_att=None, cap_gt_att=None):
        # calculate the topology loss
        if self.progressive_optimization:
            layer_num = im_dep.shape[1]
            img_topology_loss_list = []
            cap_topology_loss_list = []

            for idx in range(layer_num):
                img_topology_loss_list.append(self.compute_loss(im_dep[:, idx], gt_dep.float()))
                cap_topology_loss_list.append(self.compute_loss(cap_dep[:, idx], gt_dep.float()))

            img_topology_loss = 0
            cap_topology_loss = 0
            for idx in range(layer_num):
                img_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * img_topology_loss_list[idx]
                cap_topology_loss += (self.decay_factor ** (layer_num - 1 - idx)) * cap_topology_loss_list[idx]

        else:
            im_dep = im_dep[:, -1]
            cap_dep = cap_dep[:, -1]

            img_topology_loss = self.compute_loss(im_dep, gt_dep.float())
            cap_topology_loss = self.compute_loss(cap_dep, gt_dep.float())

        topology_loss = img_topology_loss + cap_topology_loss

        loss = topology_loss

        data = {
            "img_topology_loss": img_topology_loss,
            "cap_topology_loss": cap_topology_loss,
        }

        # calculate the contrastive loss
        if self.contrastive_loss:
            img_contrastive_loss, cap_contrastive_loss = \
                self.contrastive_supervision_loss(im_dep, cap_dep, gt_dep)
            contrastive_loss = img_contrastive_loss + cap_contrastive_loss
            loss += self.opt.contrastive_loss_weight * contrastive_loss

            data["img_contrastive_loss"] = img_contrastive_loss
            data["cap_contrastive_loss"] = cap_contrastive_loss

        # calculate the task completion loss (consider in 2nd stage)
        if self.task_completion and self.include_second_stage:
            img_completion_loss, cap_completion_loss = \
                self.task_completion_loss(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            completion_loss = img_completion_loss + cap_completion_loss

            loss += self.opt.task_completion_weight * completion_loss

            data["img_completion_loss"] = img_completion_loss
            data["cap_completion_loss"] = cap_completion_loss

        # calculate the stepwise task completion loss (consider in 2nd stage)
        if self.stepwise_task_completion and self.include_second_stage:
            im_propagation_loss, im_receive_loss, cap_propagation_loss, cap_receive_loss = \
                self.stepwise_task_completion_loss(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            stepwise_completion_loss = im_propagation_loss + im_receive_loss + cap_propagation_loss + cap_receive_loss

            loss += self.opt.stepwise_task_completion_weight * stepwise_completion_loss

            data["im_propagation_loss"] = im_propagation_loss
            data["im_receive_loss"] = im_receive_loss
            data["cap_propagation_loss"] = cap_propagation_loss
            data["cap_receive_loss"] = cap_receive_loss

        if self.attention_supervision:
            im_att_loss, cap_att_loss = \
                self.attention_loss(gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            att_loss = im_att_loss + cap_att_loss

            loss += self.opt.attention_supervision_weight * att_loss

            data["im_att_loss"] = im_att_loss
            data["cap_att_loss"] = cap_att_loss

        data["loss"] = loss

        return data

    def compute_loss(self, pred, gt):
        retrieval_loss_list = []
        dependency_loss_list = []
        if self.opt.fp16:
            eps = 1e-16
            pred = torch.log((pred + eps) / (1 - pred + eps))
            # for the retrieval task
            retrieval_from_start_loss = self.criterion(pred[:, 0], gt[:, 0])
            retrieval_to_end_loss = self.criterion(pred[:, :, 1], gt[:, :, 1])
            # for the dependency relationship
            dependency_loss = self.criterion(pred, gt)
        else:
            # for the retrieval task
            retrieval_from_start_loss = self.criterion(pred[:, 0], gt[:, 0])
            retrieval_to_end_loss = self.criterion(pred[:, :, 1], gt[:, :, 1])
            # for the dependency relationship
            dependency_loss = self.criterion(pred, gt)

        retrieval_from_start_pos_sample = (gt[:, 0].sum(-1)).int()
        retrieval_from_start_neg_sample = retrieval_from_start_pos_sample * self.ratio
        possible_max_neg_sample = gt[:, 0].shape[-1] - retrieval_from_start_pos_sample
        retrieval_from_start_neg_sample = torch.minimum(retrieval_from_start_neg_sample, possible_max_neg_sample)

        retrieval_to_end_pos_sample = (gt[:, :, 1].sum(-1)).int()
        retrieval_to_end_neg_sample = retrieval_to_end_pos_sample * self.ratio
        possible_max_neg_sample = gt[:, :, 1].shape[-1] - retrieval_to_end_pos_sample
        retrieval_to_end_neg_sample = torch.minimum(retrieval_to_end_neg_sample, possible_max_neg_sample)


        dependency_pos_sample = (gt.sum(-1).sum(-1)).int() - retrieval_from_start_pos_sample - retrieval_to_end_pos_sample
        dependency_neg_sample = dependency_pos_sample * self.ratio
        possible_max_neg_sample = (gt.shape[-1] - 1) * (gt.shape[-1] - 1) - dependency_pos_sample
        dependency_neg_sample = torch.minimum(dependency_neg_sample, possible_max_neg_sample)

        for idx in range(gt.shape[0]):
            retrieval_from_start_pos_loss = retrieval_from_start_loss[idx][gt[idx, 0] == 1]
            retrieval_from_start_neg_loss = retrieval_from_start_loss[idx][gt[idx, 0] == 0]
            retrieval_from_start_neg_loss = retrieval_from_start_neg_loss.topk(retrieval_from_start_neg_sample[idx])[0]
            retrieval_loss_list.append(retrieval_from_start_pos_loss)
            retrieval_loss_list.append(retrieval_from_start_neg_loss)

            retrieval_to_end_pos_loss = retrieval_to_end_loss[idx][gt[idx, :, 1] == 1]
            retrieval_to_end_neg_loss = retrieval_to_end_loss[idx][gt[idx, :, 1] == 0]
            retrieval_to_end_neg_loss = retrieval_to_end_neg_loss.topk(retrieval_to_end_neg_sample[idx])[0]
            retrieval_loss_list.append(retrieval_to_end_pos_loss)
            retrieval_loss_list.append(retrieval_to_end_neg_loss)

            if self.self_dependency:
                consider_idx = gt.new_zeros(gt[idx].shape)
                consider_idx[:2, :2] = 1
                consider_idx[0, :] = 1
                consider_idx[:, 1] = 1
                tmp = consider_idx[gt[idx, 0] == 1]
                tmp[:, gt[idx, 0] == 1] = 1
                consider_idx[gt[idx, 0] == 1] = tmp
                # consider_idx[2:, gt[idx, 0] == 1] = 1
                # consider_idx[gt[idx, 0] == 1, 2:] = 1
            else:
                consider_idx = gt.new_zeros(gt[idx].shape)
                consider_idx[2:, 2:] = 1
            dependency_pos_loss = dependency_loss[idx][torch.logical_and(gt[idx] == 1, consider_idx)]
            dependency_neg_loss = dependency_loss[idx][torch.logical_and(gt[idx] == 0, consider_idx)]
            if len(dependency_neg_loss) > dependency_neg_sample[idx]:
                dependency_neg_loss = dependency_neg_loss.topk(dependency_neg_sample[idx])[0]
            dependency_loss_list.append(dependency_pos_loss)
            dependency_loss_list.append(dependency_neg_loss)

        retrieval_loss = torch.cat(retrieval_loss_list, dim=0)
        dependency_loss = torch.cat(dependency_loss_list, dim=0)

        if self.include_second_stage:
            loss = torch.cat([retrieval_loss, dependency_loss], dim=0).mean()
        else:
            loss = retrieval_loss.mean()

        return loss


class TaskCompletionLoss(nn.Module):
    """
    Compute Online hard sample mining loss (binary cross entropy loss)
    """

    def __init__(self, opt):
        super(TaskCompletionLoss, self).__init__()
        self.opt = opt
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor
        if not hasattr(self.opt, 'similarity'):
            self.opt.similarity = "jsd"
        if self.opt.similarity == "jsd":
            self.similarity_function = cal_jsd_score
        elif self.opt.similarity == "sim":
            self.similarity_function = cal_sim_score
        else:
            raise "No implementation"
        self.weighted_task_completion = self.opt.weighted_task_completion

    def message_propagation(self, att_score, topological_graph_matrix, predict_topological_graph_matrix):
        eps = 1e-15
        # for prediction force the end to end edge is 1, end to other is 0
        # tmp_predict_topological_graph_matrix = predict_topological_graph_matrix.new_zeros(predict_topological_graph_matrix.shape)
        # tmp_predict_topological_graph_matrix[:, 1, 1] = 1
        # tmp_predict_topological_graph_matrix[:, 0] = predict_topological_graph_matrix[:, 0]
        # tmp_predict_topological_graph_matrix[:, 2:] = predict_topological_graph_matrix[:, 2:]
        # predict_topological_graph_matrix = tmp_predict_topological_graph_matrix

        GP = topological_graph_matrix * predict_topological_graph_matrix

        out_degree = GP.sum(-1, keepdims=True) / (predict_topological_graph_matrix.sum(-1, keepdims=True) + eps) / \
                     (topological_graph_matrix.sum(-1, keepdims=True) + eps)

        in_degree = GP.sum(1, keepdims=True) / (predict_topological_graph_matrix.sum(1, keepdims=True) + eps)

        M = out_degree * in_degree * predict_topological_graph_matrix * topological_graph_matrix
        score = att_score.new_zeros(att_score.shape[0], att_score.shape[1] + 2)
        score[:, 0] = 1
        score[:, 2:] = att_score

        final_score = []
        for idx in range(att_score.shape[0]):
            cur_score = score[idx]
            cur_M = M[idx]
            for ii in range(topological_graph_matrix[idx, 0, 2:].sum() + 1):
                cur_score = cur_M.T @ cur_score
            final_score.append(cur_score[1] / (topological_graph_matrix[idx, 0, 2:].sum() + 1))

        final_score = torch.stack(final_score)

        return final_score

    def compute_score(self, im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask):

        im_att_score = self.similarity_function(im_att, im_gt_att.unsqueeze(0).repeat(im_att.shape[0], 1, 1))
        cap_att_score = self.similarity_function(cap_att, cap_gt_att.unsqueeze(0).repeat(cap_att.shape[0], 1, 1),
                                      cap_emb_mask.unsqueeze(0).repeat(cap_att.shape[0], 1, 1))

        im_completion_score = self.message_propagation(im_att_score, gt_dep, im_dep)
        cap_completion_score = self.message_propagation(cap_att_score, gt_dep, cap_dep)

        return im_completion_score, cap_completion_score

    def forward(self, im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask):

        if self.weighted_task_completion:
            weights = gt_dep[:, 2:, 2:].sum(-1).sum(-1) + 1
        else:
            weights = gt_dep[:, 2:, 2:].sum(-1).sum(-1) * 0 + 1
        weights = weights / weights.sum()

        if self.progressive_optimization:
            layer_num = im_att.shape[1]

            img_completion_score_list = []
            cap_completion_score_list = []

            for idx in range(layer_num):
                img_completion_score, cap_completion_score = \
                    self.compute_score(im_dep[:, idx], cap_dep[:, idx], gt_dep, im_att[:, idx], cap_att[:, idx], im_gt_att, cap_gt_att, cap_emb_mask)

                img_completion_score_list.append(img_completion_score * weights)
                cap_completion_score_list.append(cap_completion_score * weights)

            img_completion_loss = 0
            cap_completion_loss = 0
            for idx in range(layer_num):
                img_completion_loss -= (self.decay_factor ** (layer_num - 1 - idx)) * img_completion_score_list[
                    idx].sum()
                cap_completion_loss -= (self.decay_factor ** (layer_num - 1 - idx)) * cap_completion_score_list[
                    idx].sum()

        else:
            im_att = im_att[:, -1]
            cap_att = cap_att[:, -1]
            img_completion_score, cap_completion_score = \
                self.compute_score(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            img_completion_loss, cap_completion_loss = -(img_completion_score * weights).sum(), -(cap_completion_score * weights).sum()

        return img_completion_loss,  cap_completion_loss


class StepwiseTaskCompletionLoss(nn.Module):
    """
    Stepwise Task Completion Loss loss
    """

    def __init__(self, opt):
        super(StepwiseTaskCompletionLoss, self).__init__()
        self.opt = opt
        self.progressive_optimization = self.opt.progressive_optimization
        self.decay_factor = self.opt.decay_factor
        if not hasattr(self.opt, 'similarity'):
            self.opt.similarity = "jsd"
        if self.opt.similarity == "jsd":
            self.similarity_function = cal_jsd_score
        elif self.opt.similarity == "sim":
            self.similarity_function = cal_sim_score
        else:
            raise "No implementation"
        self.weighted_task_completion = self.opt.weighted_task_completion

    def message_propagation_to(self, att_score, topological_graph_matrix, predict_topological_graph_matrix):
        eps = 1e-15

        GP = topological_graph_matrix * predict_topological_graph_matrix

        out_degree = GP.sum(-1, keepdims=True) / (predict_topological_graph_matrix.sum(-1, keepdims=True) + eps)
        information_dist = GP / (topological_graph_matrix.sum(-1, keepdims=True) + eps)

        M = out_degree * information_dist
        score = att_score.new_zeros(att_score.shape[0], att_score.shape[1] + 2)
        score[:, 0] = 1
        score[:, 2:] = att_score

        final_score = []
        for idx in range(att_score.shape[0]):
            cur_score = score[idx]
            cur_M = M[idx]
            propagation_score = cur_M.T @ cur_score
            out_nodes = topological_graph_matrix[idx, 0].clone()
            out_nodes[1] = 1
            out_avg_score = propagation_score[out_nodes==1].mean()
            final_score.append(out_avg_score)
        final_score = torch.stack(final_score)
        return final_score

    def message_propagation_from(self, att_score, topological_graph_matrix, predict_topological_graph_matrix):
        eps = 1e-15

        GP = topological_graph_matrix * predict_topological_graph_matrix

        information_dist = GP / (topological_graph_matrix.sum(-1, keepdims=True) + eps)
        in_degree = GP.sum(1, keepdims=True) / (predict_topological_graph_matrix.sum(1, keepdims=True) + eps)

        M = information_dist * in_degree
        score = att_score.new_zeros(att_score.shape[0], att_score.shape[1] + 2)
        score[:, 0] = 1
        score[:, 2:] = att_score

        final_score = []
        for idx in range(att_score.shape[0]):
            cur_score = score[idx]
            cur_M = M[idx]
            propagation_score = cur_M.T @ cur_score
            out_nodes = topological_graph_matrix[idx, 0].clone()
            out_nodes[1] = 1
            out_avg_score = propagation_score[out_nodes == 1].mean()
            final_score.append(out_avg_score)
        final_score = torch.stack(final_score)

        return final_score

    def compute_score(self, im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask):

        im_att_score = self.similarity_function(im_att, im_gt_att.unsqueeze(0).repeat(im_att.shape[0], 1, 1))
        cap_att_score = self.similarity_function(cap_att, cap_gt_att.unsqueeze(0).repeat(cap_att.shape[0], 1, 1),
                                      cap_emb_mask.unsqueeze(0).repeat(cap_att.shape[0], 1, 1))

        im_propagation_score = self.message_propagation_to(im_att_score, gt_dep, im_dep)
        im_receive_score = self.message_propagation_from(im_att_score, gt_dep, im_dep)
        cap_propagation_score = self.message_propagation_to(cap_att_score, gt_dep, cap_dep)
        cap_receive_score = self.message_propagation_from(cap_att_score, gt_dep, cap_dep)

        return im_propagation_score, im_receive_score, cap_propagation_score, cap_receive_score

    def forward(self, im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask):

        if self.weighted_task_completion:
            weights = gt_dep[:, 2:, 2:].sum(-1).sum(-1) + 1
        else:
            weights = gt_dep[:, 2:, 2:].sum(-1).sum(-1) * 0 + 1
        weights = weights / weights.sum()

        if self.progressive_optimization:
            layer_num = im_att.shape[1]

            img_propagation_score_list = []
            img_receive_score_list = []
            cap_propagation_score_list = []
            cap_receive_score_list = []

            for idx in range(layer_num):
                im_propagation_score, im_receive_score, cap_propagation_score, cap_receive_score = \
                    self.compute_score(im_dep[:, idx], cap_dep[:, idx], gt_dep, im_att[:, idx], cap_att[:, idx], im_gt_att, cap_gt_att, cap_emb_mask)

                img_propagation_score_list.append(im_propagation_score * weights)
                img_receive_score_list.append(im_receive_score * weights)
                cap_propagation_score_list.append(cap_propagation_score * weights)
                cap_receive_score_list.append(cap_receive_score * weights)

            img_propagation_loss = 0
            img_receive_loss = 0
            cap_propagation_loss = 0
            cap_receive_loss = 0
            for idx in range(layer_num):
                img_propagation_loss -= (self.decay_factor ** (layer_num - 1 - idx)) * img_propagation_score_list[
                    idx].sum()
                img_receive_loss -= (self.decay_factor ** (layer_num - 1 - idx)) * img_receive_score_list[
                    idx].sum()
                cap_propagation_loss -= (self.decay_factor ** (layer_num - 1 - idx)) * cap_propagation_score_list[
                    idx].sum()
                cap_receive_loss -= (self.decay_factor ** (layer_num - 1 - idx)) * cap_receive_score_list[
                    idx].sum()

        else:
            im_att = im_att[:, -1]
            cap_att = cap_att[:, -1]
            im_propagation_score, im_receive_score, cap_propagation_score, cap_receive_score = \
                self.compute_score(im_dep, cap_dep, gt_dep, im_att, cap_att, im_gt_att, cap_gt_att, cap_emb_mask)
            img_propagation_loss, img_receive_loss, cap_propagation_loss, cap_receive_loss = \
                -(im_propagation_score * weights).sum(), -(im_receive_score * weights).sum(), \
                -(cap_propagation_score * weights).sum(), -(cap_receive_score * weights).sum()

        return img_propagation_loss, img_receive_loss, cap_propagation_loss, cap_receive_loss


