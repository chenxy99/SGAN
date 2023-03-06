import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
import json
import os.path as osp
import imageio.v2 as imageio
import random
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import html
import torch.nn.functional as F
import os
import torch

from lib.datasets.process import process_caption, process_goal

import logging

logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class RawImageDataset(Dataset):

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.opt = opt
        self.extend_graph = self.opt.extend_graph
        self.dependency_type = self.opt.dependency_type
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.data_split = data_split
        self.max_goal_length = opt.max_goal_length
        self.max_problem_solving_step = opt.max_problem_solving_step

        loc_info = osp.join(data_path, "final_result_1_29_update.json")
        # get wikihow data information
        with open(loc_info) as f:
            data = json.load(f)
        self.data = []
        for key, value in data.items():
            if value["split"] == data_split and value["dependency_type"] in self.dependency_type:
                value["goal_id"] = key
                self.data.append(value)

        if (data_split == "train" or data_split == "val") and self.opt.tiny:
            self.data = self.data[:1000]

        if (data_split == "val" or data_split == "test") and self.opt.interleave_validation:
            self.data = self.data[:(len(self.data) // 8) * 8]
            # interleave of the data
            tmp = []
            for idx in range(8):
                tmp.extend(self.data[idx::8])
            self.data = tmp

        # construct data structure
        self.captions = []
        self.tasks = []
        self.methods = []
        self.task_tokens = []
        self.method_tokens = []
        self.image_paths = []
        self.hit_ids = []
        self.caption_attention_index = []
        self.caption_attention_word = []
        self.bbox_info = []
        self.dependency_type = []
        self.topological_graph = []
        for value in tqdm(self.data):
            self.captions.append(value["step_list"])
            self.tasks.append(value["task_title"])
            self.methods.append(value["method_title"])
            self.image_paths.append(["/".join(_.split("/")[-2:]) for _ in value["image_url"]])
            self.hit_ids.append(value["post_id"] + "_" + value["method_idx"])
            self.caption_attention_index.append(value["step_to_object_selected_index_position_result"])
            self.caption_attention_word.append(value["step_to_object_selected_result"])
            self.bbox_info.append(value["step_to_object_bbox_result"])
            self.dependency_type.append(value["dependency_type"])
            self.topological_graph.append(value["step_to_dependency_index_result"])

        if "wikihow" in data_name:
            self.image_base = osp.join(data_path, 'images')

        assert len(self.captions) == len(self.tasks) and len(self.captions) == len(self.methods), \
            "The lengths of captions, tasks and methods are not equal!"

        # Set related parameters according to the pre-trained backbone **
        assert 'backbone' in opt.precomp_enc_type

        self.backbone_source = opt.backbone_source
        self.base_target_size = 256
        self.crop_ratio = 0.875
        self.train_scale_rate = 1
        if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
            self.base_target_size = int(self.base_target_size * opt.input_scale_factor)
            logger.info('Input images are scaled by factor {}'.format(opt.input_scale_factor))
        if 'detector' in self.backbone_source:
            self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        self.length = len(self.data)

        self.image_num = 0
        self.image_path_to_idx = {}
        counter = 0
        for value in self.image_paths:
            self.image_num += len(value)
            for image_path in value:
                self.image_path_to_idx[image_path] = counter
                counter += 1

        self.cap_num = self.image_num


    def __getitem__(self, index):
        goal_index = self.hit_ids[index]
        caption = self.captions[index]
        cap_att_indexes = self.caption_attention_index[index]
        cap_att_words = self.caption_attention_word[index]
        dependency_type = self.dependency_type[index]

        # grounded attention indicator for pair image and caption attention
        grounded_attention_indicator = np.zeros((self.opt.max_problem_solving_step, self.opt.attention_step_num), dtype=np.float32)

        # sample possible sequence
        topological_graph = self.topological_graph[index]
        if self.opt.extend_graph:
            extended_topological_graph = []
            for value in topological_graph:
                cur_list = []
                for idx in value:
                    self.find_complete_dependency(cur_list, idx, topological_graph)
                cur_list = list(set(cur_list))
                extended_topological_graph.append(sorted(cur_list))
            topological_graph = extended_topological_graph

        # topological_graph matrix form with the goal as the 0 index
        # topological_graph_matrix[i, j] represents a link from node i to node j
        # manually add start point [idx=0] and end point [idx=1]
        # topological_graph_matrix = np.zeros((len(topological_graph) + 2, len(topological_graph) + 2), dtype=np.float32)
        topological_graph_matrix = np.zeros((self.max_problem_solving_step + 2, self.max_problem_solving_step + 2), dtype=np.float32)
        for idx, dependency in enumerate(topological_graph):
            if len(dependency) == 0:
                topological_graph_matrix[0][idx + 2] = 1
            else:
                topological_graph_matrix[0][idx + 2] = 1
                for value in dependency:
                    topological_graph_matrix[value + 2][idx + 2] = 1
        # topological_graph_matrix[topological_graph_matrix.sum(-1) == 0, -1] = 1
        for idx in range(len(topological_graph)):
            if topological_graph_matrix[idx + 2].sum() == 0:
                topological_graph_matrix[idx + 2, 1] = 1
        topological_graph_matrix[1, 1] = 1

        # for the attention in captions
        caption_steps, grounded_attentions, non_grounded_attentions = [], [], []
        aggregate_caption_attentions = []
        grounded_word2idx_list = []
        for idx in range(len(caption)):
            # get the caption attention index
            cap_att_index = cap_att_indexes[idx]

            caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption[idx])
            concatenated_tokens = "".join(caption_tokens)
            concatenated_tokens_start_end_pt = []
            start_pt = 0
            end_pt = 0
            for _ in caption_tokens:
                end_pt += len(_)
                concatenated_tokens_start_end_pt.append((start_pt, end_pt))
                start_pt = end_pt

            grounded_attention_index = cap_att_index[0]
            non_grounded_attention_index = cap_att_index[1]

            grounded_attention = [[0] * len(caption_tokens) for _ in range(self.opt.attention_step_num)]
            non_grounded_attention = [0] * len(caption_tokens)
            # for the grounded annotation
            grounded_word2idx = {}
            for ii, (start_pos, end_pos) in enumerate(grounded_attention_index):
                if ii >= self.opt.attention_step_num:
                    break
                grounded_attention_indicator[idx, ii] = 1
                selected_words = caption[idx][start_pos:end_pos]
                grounded_word2idx[selected_words.lower()] = ii
                selected_word_tokens = self.tokenizer.basic_tokenizer.tokenize(selected_words)
                concatenated_selected_word_tokens = "".join(selected_word_tokens)
                concatenated_start_pos = concatenated_tokens.find(concatenated_selected_word_tokens)
                concatenated_end_pos = concatenated_start_pos + len(concatenated_selected_word_tokens)
                # annotate the attention label in grounded_attention
                flag = 0
                for i, (cap_start_pos, cap_end_pos) in enumerate(concatenated_tokens_start_end_pt):
                    if cap_start_pos == concatenated_start_pos:
                        flag = 1
                    if flag == 1:
                        grounded_attention[ii][i] = flag
                    if cap_end_pos == concatenated_end_pos:
                        flag = 0
            grounded_word2idx_list.append(grounded_word2idx)
            # for the non-grounded annotation
            for start_pos, end_pos in non_grounded_attention_index:
                selected_words = caption[idx][start_pos:end_pos]
                selected_word_tokens = self.tokenizer.basic_tokenizer.tokenize(selected_words)
                concatenated_selected_word_tokens = "".join(selected_word_tokens)
                concatenated_start_pos = concatenated_tokens.find(concatenated_selected_word_tokens)
                concatenated_end_pos = concatenated_start_pos + len(concatenated_selected_word_tokens)
                flag = 0
                for i, (cap_start_pos, cap_end_pos) in enumerate(concatenated_tokens_start_end_pt):
                    if cap_start_pos == concatenated_start_pos:
                        flag = 1
                    if flag == 1:
                        non_grounded_attention[i] = flag
                    if cap_end_pos == concatenated_end_pos:
                        flag = 0

            # Convert caption (string) to word ids (with Size Augmentation at training time).
            target, target_grounded_attention, target_non_grounded_attention = \
                process_caption(self.tokenizer, caption_tokens, grounded_attention, non_grounded_attention, self.train)
            caption_steps.append(target)
            grounded_attentions.append(target_grounded_attention)
            non_grounded_attentions.append(target_non_grounded_attention.unsqueeze(0))

            aggregate_caption_attention = ((target_grounded_attention.sum(0) + target_non_grounded_attention) > 0).float().unsqueeze(0)
            aggregate_caption_attentions.append(aggregate_caption_attention)


        # handle image
        image_paths = self.image_paths[index]
        bbox_info = self.bbox_info[index]

        images = []
        image_index = []
        image_ids = []
        grounded_attention_maps, non_grounded_attention_maps = [], []
        scale_down_grounded_attention_maps, scale_down_non_grounded_attention_maps = [], []
        aggregate_attention_maps, scale_down_aggregate_attention_maps = [], []
        for idx in range(len(image_paths)):
            image_ids.append(image_paths[idx].split("/")[-1])
            image_index.append(self.image_path_to_idx[image_paths[idx]])
            image_path = os.path.join(self.image_base, image_paths[idx])
            im_in = np.array(imageio.imread(image_path))
            im_grounded_attention_map = np.zeros((im_in.shape[0], im_in.shape[1], grounded_attentions[idx].shape[0]), dtype=np.float32)
            im_non_grounded_attention_map = np.zeros((im_in.shape[0], im_in.shape[1], 1), dtype=np.float32)
            curr_bbox_info = bbox_info[idx]
            # for the grounded annotation
            grounded_curr_bbox_info = curr_bbox_info[0]
            grounded_word2idx = grounded_word2idx_list[idx]
            for objs_bbox_info in grounded_curr_bbox_info:
                if objs_bbox_info is not None:
                    for obj_bbox_info in objs_bbox_info:
                        left = obj_bbox_info["left"]
                        top = obj_bbox_info["top"]
                        width = obj_bbox_info["width"]
                        height = obj_bbox_info["height"]
                        object_label = obj_bbox_info["label"]
                        wordidx = grounded_word2idx.get(html.unescape(object_label), None)
                        if wordidx is None:
                            continue
                        im_grounded_attention_map[top:top + height, left:left + width, wordidx] = 1
            # for the non-grounded annotation
            important_curr_bbox_info = curr_bbox_info[2]
            for objs_bbox_info in important_curr_bbox_info:
                if objs_bbox_info is not None:
                    for obj_bbox_info in objs_bbox_info:
                        left = obj_bbox_info["left"]
                        top = obj_bbox_info["top"]
                        width = obj_bbox_info["width"]
                        height = obj_bbox_info["height"]
                        im_non_grounded_attention_map[top:top + height, left:left + width] = 1
            processed_image, processed_im_grounded_attention_map, processed_im_non_grounded_attention_map = \
                self._process_image(im_in, im_grounded_attention_map, im_non_grounded_attention_map)
            image = torch.Tensor(processed_image)
            image = image.permute(2, 0, 1).contiguous()
            im_grounded_attention_map = torch.Tensor(processed_im_grounded_attention_map)
            im_grounded_attention_map = im_grounded_attention_map.permute(2, 0, 1).contiguous()
            im_non_grounded_attention_map = torch.Tensor(processed_im_non_grounded_attention_map[:, :, np.newaxis])
            im_non_grounded_attention_map = im_non_grounded_attention_map.permute(2, 0, 1).contiguous()
            aggregate_grounded_image_attention = ((im_grounded_attention_map.sum(0) + im_non_grounded_attention_map[0]) > 0).float().unsqueeze(0)
            images.append(image)
            grounded_attention_maps.append(im_grounded_attention_map)
            non_grounded_attention_maps.append(im_non_grounded_attention_map)
            aggregate_attention_maps.append(aggregate_grounded_image_attention)

            # scale down to the high level feature map
            if self.train:
                target_size = self.base_target_size * self.train_scale_rate
            else:
                target_size = self.base_target_size
            grounded_image_attention = im_grounded_attention_map
            grounded_image_attention = F.interpolate(grounded_image_attention.unsqueeze(1), mode='bilinear',
                                                     size=(target_size // 32, target_size // 32)).squeeze(1)
            non_grounded_image_attention = im_non_grounded_attention_map
            non_grounded_image_attention = F.interpolate(non_grounded_image_attention.unsqueeze(1), mode='bilinear',
                                                     size=(target_size // 32, target_size // 32)).squeeze(1)
            scale_down_aggregate_grounded_image_attention = ((grounded_image_attention.sum(0) + non_grounded_image_attention[0]) > 0).float().unsqueeze(0)
            scale_down_grounded_attention_maps.append(grounded_image_attention)
            scale_down_non_grounded_attention_maps.append(non_grounded_image_attention)
            scale_down_aggregate_attention_maps.append(scale_down_aggregate_grounded_image_attention)


        # get the goal tokens
        task = self.tasks[index]
        task_tokens = self.tokenizer.basic_tokenizer.tokenize(task)
        method = self.methods[index]
        method_tokens = self.tokenizer.basic_tokenizer.tokenize(method)

        # Convert task (string) and method (string) to word ids (with Size Augmentation at training time)
        goal = process_goal(self.tokenizer, task_tokens, method_tokens, self.train)


        dummy_image = images[0] * 0
        for i in range(3):
            dummy_image[i] += self.imagenet_mean[i]
        dummy_grounded_attention = [[0] * 0 for _ in range(self.opt.attention_step_num)]
        dummy_non_grounded_attention = [0] * 0
        dummy_caption, dummy_grounded_attention, dummy_non_grounded_attention = \
            process_caption(self.tokenizer, [], dummy_grounded_attention, dummy_non_grounded_attention, self.train)

        # append the dummy image/caption to a fix problem-solving steps
        for idx in range(len(image_paths), self.max_problem_solving_step):
            caption_steps.append(dummy_caption)
            grounded_attentions.append(dummy_grounded_attention)
            non_grounded_attentions.append(dummy_non_grounded_attention)
            aggregate_caption_attentions.append(dummy_non_grounded_attention)

            images.append(dummy_image)
            scale_down_grounded_attention_maps.append(scale_down_grounded_attention_maps[0] * 0)
            scale_down_non_grounded_attention_maps.append(scale_down_non_grounded_attention_maps[0] * 0)
            scale_down_aggregate_attention_maps.append(scale_down_aggregate_attention_maps[0] * 0)

        actual_problem_solving_step_indicator = [1] * len(image_paths) + [0] * (self.max_problem_solving_step - len(image_paths))
        dataset_idx = index

        data = {}
        data["goal"] = goal
        data["images"] = images
        data["caption_steps"] = caption_steps
        data["grounded_attentions"] = grounded_attentions
        data["non_grounded_attentions"] = non_grounded_attentions
        data["aggregate_caption_attentions"] = aggregate_caption_attentions
        data["scale_down_grounded_attention_maps"] = scale_down_grounded_attention_maps
        data["scale_down_non_grounded_attention_maps"] = scale_down_non_grounded_attention_maps
        data["scale_down_aggregate_attention_maps"] = scale_down_aggregate_attention_maps
        data["actual_problem_solving_step_indicator"] = actual_problem_solving_step_indicator
        data["topological_graph_matrix"] = topological_graph_matrix
        data["information"] = {
            "goal_index": goal_index,
            "task": task,
            "method": method,
            "image_index": image_index,
            "image_ids": image_ids,
            "dataset_idx": dataset_idx,
            "caption" : caption,
            "cap_att_indexes": cap_att_indexes,
            "cap_att_words": cap_att_words,
            "bbox_info": bbox_info,
            "topological_graph": topological_graph,
            "dependency_type": dependency_type
        }
        return data

    def find_complete_dependency(self, cur_list, search_idx, graph):
        cur_list.append(search_idx)
        for idx in graph[search_idx]:
            self.find_complete_dependency(cur_list, idx, graph)

    def __len__(self):
        return self.length

    def _sample_possible_sequence(self, topological_graph):
        sampled_sequence = []
        dependency_step = {i: set(value) for i, value in enumerate(topological_graph)}

        for idx in range(len(topological_graph)):
            possible_step = []
            for index in range(len(topological_graph)):
                if len(dependency_step[index].intersection(sampled_sequence)) == len(dependency_step[index]) and index not in sampled_sequence:
                    # complete all the pre-step
                    possible_step.append(index)
            sampled_sequence.append(random.choice(possible_step))

        return sampled_sequence

    def _process_image(self, im_in, im_g_att_in, im_att_in):
        """
            Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation.
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        if 'detector' in self.backbone_source:
            im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)

        if self.train:
            target_size = self.base_target_size * self.train_scale_rate
        else:
            target_size = self.base_target_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im, x_start, y_start = self._crop(im, crop_size_h, crop_size_w, random=True)
            processed_im_g_att_in = im_g_att_in[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]
            processed_im_att_in = im_att_in[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]
        else:
            processed_im = im
            processed_im_g_att_in = im_g_att_in
            processed_im_att_in = im_att_in

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)
        processed_im_g_att_in = cv2.resize(processed_im_g_att_in, None, None, fx=im_scale_x, fy=im_scale_y,
                                           interpolation=cv2.INTER_LINEAR)
        processed_im_att_in = cv2.resize(processed_im_att_in, None, None, fx=im_scale_x, fy=im_scale_y,
                                         interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)
                processed_im_g_att_in = self._hori_flip(processed_im_g_att_in)
                processed_im_att_in = self._hori_flip(processed_im_att_in)

        # Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
        else:
            processed_im = self._imagenet_norm(processed_im)

        # visualize the processed image and the attention map
        # _im_show(processed_im)
        # _im_show(processed_im_g_att_in[:, :, 0])
        # _im_show(processed_im_g_att_in[:, :, 1])
        # _im_show(processed_im_att_in)

        return processed_im, processed_im_g_att_in, processed_im_att_in

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    def _detector_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im, x_start, y_start

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im

    def collate_fn(self, batch):
        goal_batch = []
        images_batch = []
        caption_steps_batch = []
        grounded_attentions_batch = []
        non_grounded_attentions_batch = []
        aggregate_caption_attentions_batch = []
        scale_down_grounded_attention_maps_batch = []
        scale_down_non_grounded_attention_maps_batch = []
        scale_down_aggregate_attention_maps_batch = []
        actual_problem_solving_step_indicator_batch = []
        topological_graph_matrix_batch = []
        information_batch = []

        batch_size = len(batch)

        for sample in batch:
            tmp_goal, tmp_images, tmp_caption_steps, \
            tmp_grounded_attentions, tmp_non_grounded_attentions, tmp_aggregate_caption_attentions, \
            tmp_scale_down_grounded_attention_maps, tmp_scale_down_non_grounded_attention_maps, tmp_scale_down_aggregate_attention_maps, \
            tmp_actual_problem_solving_step_indicator, tmp_topological_graph_matrix, tmp_information \
                = sample["goal"], sample["images"], sample["caption_steps"], \
                  sample["grounded_attentions"], sample["non_grounded_attentions"], sample["aggregate_caption_attentions"], \
                  sample["scale_down_grounded_attention_maps"], sample["scale_down_non_grounded_attention_maps"], sample["scale_down_aggregate_attention_maps"], \
                  sample["actual_problem_solving_step_indicator"], sample["topological_graph_matrix"], sample["information"]

            goal_batch.append(tmp_goal)
            images_batch.append(tmp_images)
            caption_steps_batch.append(tmp_caption_steps)
            grounded_attentions_batch.append(tmp_grounded_attentions)
            non_grounded_attentions_batch.append(tmp_non_grounded_attentions)
            aggregate_caption_attentions_batch.append(tmp_aggregate_caption_attentions)
            scale_down_grounded_attention_maps_batch.append(tmp_scale_down_grounded_attention_maps)
            scale_down_non_grounded_attention_maps_batch.append(tmp_scale_down_non_grounded_attention_maps)
            scale_down_aggregate_attention_maps_batch.append(tmp_scale_down_aggregate_attention_maps)
            actual_problem_solving_step_indicator_batch.append(tmp_actual_problem_solving_step_indicator)
            topological_graph_matrix_batch.append(tmp_topological_graph_matrix)
            information_batch.append(tmp_information)


        # Merge goals (convert tuple of 1D tensor to 2D tensor)
        goal_list = []
        for value in goal_batch:
            goal_list.append(value)
        goal_lengths = [len(goal) for goal in goal_list]
        goal_targets = torch.zeros(len(goal_list), max(goal_lengths)).long()
        for i, goal in enumerate(goal_list):
            end = goal_lengths[i]
            goal_targets[i, :end] = goal[:end]

        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        caption_steps_list = []
        for value in caption_steps_batch:
            caption_steps_list.extend(value)
        cap_lengths = [len(cap) for cap in caption_steps_list]
        caption_steps_targets = torch.zeros(len(caption_steps_list), max(cap_lengths)).long()
        for i, cap in enumerate(caption_steps_list):
            end = cap_lengths[i]
            caption_steps_targets[i, :end] = cap[:end]
        caption_steps_targets = caption_steps_targets.view(batch_size, -1, caption_steps_targets.shape[-1])

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images_list = []
        for value in images_batch:
            images_list.extend(value)
        images_targets = torch.stack(images_list, 0)
        images_targets = images_targets.view(batch_size, -1, images_targets.shape[-3], images_targets.shape[-2], images_targets.shape[-1])

        # Merge aggregate_caption_attentions (convert tuple of 2D tensor to 3D tensor)
        aggregate_caption_attentions_list = []
        for value in aggregate_caption_attentions_batch:
            aggregate_caption_attentions_list.extend(value)
        aggregate_caption_attentions_targets = torch.zeros(len(caption_steps_list), aggregate_caption_attentions_list[0].shape[0], max(cap_lengths)).float()
        for i, att_map in enumerate(aggregate_caption_attentions_list):
            end = cap_lengths[i]
            aggregate_caption_attentions_targets[i, :, :end] = att_map
        aggregate_caption_attentions_targets = aggregate_caption_attentions_targets.view(
            batch_size, -1, aggregate_caption_attentions_targets.shape[-2], aggregate_caption_attentions_targets.shape[-1])

        # Merge scale_down_aggregate_attention_maps (convert tuple of 3D tensor to 4D tensor)
        scale_down_aggregate_attention_maps_list = []
        for value in scale_down_aggregate_attention_maps_batch:
            scale_down_aggregate_attention_maps_list.extend(value)
        scale_down_aggregate_attention_maps_targets = torch.stack(scale_down_aggregate_attention_maps_list)
        scale_down_aggregate_attention_maps_targets = scale_down_aggregate_attention_maps_targets.view(
            batch_size, -1, scale_down_aggregate_attention_maps_targets.shape[-3],
            scale_down_aggregate_attention_maps_targets.shape[-2], scale_down_aggregate_attention_maps_targets.shape[-1])

        # Merge actual_problem_solving_step_indicator (convert tuple of 1D tensor to 2D tensor)
        actual_problem_solving_step_indicator_targets = torch.tensor(actual_problem_solving_step_indicator_batch, dtype=torch.long)

        # Merge topological_graph_matrix (convert tuple of 2D tensor to 3D tensor)
        topological_graph_matrix_targets = torch.tensor(topological_graph_matrix_batch,
                                                                     dtype=torch.long)

        data = dict()
        data["goal"] = goal_targets
        data["caption_steps"] = caption_steps_targets
        data["images"] = images_targets
        data["aggregate_caption_attentions"] = aggregate_caption_attentions_targets
        data["scale_down_aggregate_attention_maps"] = scale_down_aggregate_attention_maps_targets
        data["actual_problem_solving_step_indicator"] = actual_problem_solving_step_indicator_targets
        data["topological_graph_matrix"] = topological_graph_matrix_targets
        data["information"] = information_batch

        return data


def _im_show(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

def get_loader(data_path, data_name, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True, drop_last=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = RawImageDataset(data_path, data_name, data_split, tokenizer, opt, train)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last=drop_last,
                                              collate_fn=dset.collate_fn)
    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt):
    # The batch not equal to the predefined batch size will be dropped
    if torch.cuda.is_available():
        val_per_gpu_batch_size = batch_size // torch.cuda.device_count()
    else:
        val_per_gpu_batch_size = batch_size
    assert val_per_gpu_batch_size == 8, "Need to make sure each gpu contains the batch with size 8!"
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, True, workers, train=True, drop_last=True)
    val_loader = get_loader(data_path, data_name, 'val', tokenizer, opt,
                            batch_size, False, workers, train=False, drop_last=True)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False, drop_last=True)
    return test_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data/wikihow',
                        help='path to datasets')
    parser.add_argument('--batch_size', default=24, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--max_goal_length', default=20, type=int,
                        help='Maximum of goal length.')
    parser.add_argument('--max_problem_solving_step', default=10, type=int,
                        help='Maximum of problem solving steps.')
    parser.add_argument('--attention_step_num', type=int, default=10,
                        help='The number of the multimodal attention step in a single problem solving step')
    parser.add_argument('--tiny', action='store_false',
                        help='Use the tiny training set for verify.')
    parser.add_argument('--precomp_enc_type', default="backbone",
                        help='basic|backbone')
    parser.add_argument('--backbone_path', type=str, default='',
                        help='path to the pre-trained backbone net')
    parser.add_argument('--backbone_source', type=str, default='wsl',
                        help='the source of the backbone model, detector|imagenet')
    parser.add_argument('--input_scale_factor', type=float, default=2,
                        help='The factor for scaling the input image')
    parser.add_argument('--extend_graph', action='store_true', default=True,
                        help='Whether to extend the graph.')


    opt = parser.parse_args()

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab

    train_loader, val_loader = get_loaders(data_path=opt.data_path, data_name="wikihow", tokenizer=tokenizer,
                                           batch_size=opt.batch_size, workers=0, opt=opt)

    test_loader = get_test_loader("test", data_name="wikihow", tokenizer=tokenizer, batch_size=opt.batch_size,
                                  workers=0, opt=opt)

    sample = test_loader.dataset[12]

    for batch in tqdm(train_loader):
        pass
