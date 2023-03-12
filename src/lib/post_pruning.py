import json
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
import pickle

def iterateGraph(graph, i, addToList):
	all_list = []
	if addToList:
		all_list.append(i)
	indices = graph[i]
	for idx in indices:
		all_list += iterateGraph(graph, idx, True)
	return all_list

def getPrunedGraph(graph):
	new_graph = graph.copy()
	for i in range(len(graph)):
		g = graph[i]
		l = []
		for idx in g:
			l += iterateGraph(graph, idx, False)
		new_graph[i] = [ge for ge in g if ge not in l]

	return new_graph

def simplify_graph(graph_matrix):
    _graph_matrix = graph_matrix.copy()
    _graph_matrix[0] = 0
    _graph_matrix[:, 1] = 0
    dep_graph = [np.where(_graph_matrix[:, _] == 1)[0].tolist() for _ in range(graph_matrix.shape[0])]
    simplified_dep_graph = getPrunedGraph(dep_graph)

    simplify_graph_matrix = graph_matrix * 0
    for idx in range(len(simplified_dep_graph)):
        cur_dep = simplified_dep_graph[idx]
        if len(cur_dep) > 0:
            simplify_graph_matrix[cur_dep, idx] = 1
    simplify_graph_matrix[0] = graph_matrix[0]
    simplify_graph_matrix[:, 1] = graph_matrix[:, 1]
    # remove loop
    simplify_graph_matrix[1, 1] = 0

    return simplify_graph_matrix

def post_pruning(test_case, dep_threshold=0.5, ret_threshold=0.45):
    selected_idx = np.where(test_case[0] >= ret_threshold)[0]

    # filter by retrieval result
    gt_step_with_end_idx = np.concatenate([[1], selected_idx])
    gt_step_with_start_idx = np.concatenate([[0], selected_idx])
    selection_mask = np.zeros(test_case.shape, dtype=np.float32)
    selection_mask[gt_step_with_start_idx] += 1
    selection_mask[:, gt_step_with_end_idx] += 1
    selection_mask = (selection_mask >= 2).astype(np.float32)
    selection_mask[1, 1] = 1
    selection_mask[0, 1] = 0

    test_case = test_case * selection_mask

    # remove syms
    selected_matrix = test_case[selected_idx][:,selected_idx]
    selected_matrix = selected_matrix - np.transpose(selected_matrix)
    selected_matrix[selected_matrix < 0] = 0

    selection_mask = np.zeros(test_case.shape, dtype=np.float32)
    selection_mask[selected_idx] += 1
    selection_mask[:, selected_idx] += 1
    selection_mask = (selection_mask >= 2).astype(np.float32)
    test_case[selection_mask > 0] = selected_matrix.reshape(-1)

    # remove the to end node with max prob
    if selected_matrix.shape[0] != 0:
        max_prob = selected_matrix.max(-1)
        test_case[selected_idx, 1] = test_case[selected_idx, 1]  - max_prob
    else:
        pass


    node_info = {}
    for idx in range(test_case.shape[0]):
        node_info[idx] = {
            "source": [],
            "target": []
        }
    node_info[0]["target"].append(1)
    node_info[1]["source"].append(0)

    final_link = [
        (0, 1)
    ]
    # final_link.remove((0, 1))
    used_node = [0, 1]

    # find all possible link
    possible_link = []
    for index in used_node:
        for k in node_info[index]["source"]:
            possible_link.append((k, index))

    for iter in range(len(selected_idx)):
        candidate_info = {}
        best_outer_idx = -1
        best_outer_score = -1
        best_outer_link_candidate_selected = (0, 1)
        for idx in selected_idx:
            if idx not in used_node:
                score_candidate = []
                link_candidate = []
                # go through possible link
                for value in possible_link:
                    source = value[0]
                    target = value[1]

                    from_parent =  test_case[node_info[source]["source"], idx].sum()
                    to_child = test_case[idx, node_info[target]["target"]].sum()

                    from_source = test_case[source, idx]
                    to_target = test_case[idx, target]
                    add_score = from_parent + to_child
                    add_score += from_source
                    if target == 1:
                        # need to remove test_case[source, target], and to_child miss consider M[idx, 1]
                        add_score = add_score + test_case[idx, 1] - test_case[source, 1] # target == 1, add less M[idx, 1]
                    else:
                        # to_child has M[idx, 1], need to remove
                        add_score = add_score - test_case[idx, 1] + to_target
                    score_candidate.append(add_score)
                    link_candidate.append(value)
                best_idx = np.argmax(score_candidate)
                best_value = score_candidate[best_idx]
                link_candidate_selected = link_candidate[best_idx]
                candidate_info[idx] = {
                    "best_value": best_value,
                    "link_candidate_selected": link_candidate_selected
                }
                if best_outer_score < best_value:
                    best_outer_score = best_value
                    best_outer_idx = idx
                    best_outer_link_candidate_selected = link_candidate_selected

        # replace idx and link_candidate_selected
        if best_outer_score < dep_threshold:
            continue

        idx = best_outer_idx
        link_candidate_selected = best_outer_link_candidate_selected

        used_node.append(idx)
        source = link_candidate_selected[0]
        target = link_candidate_selected[1]

        node_info[idx]["source"] = list(set(node_info[source]["source"] + [source]))
        node_info[idx]["target"] = list(set(node_info[target]["target"] + [target]))
        node_info[source]["target"].append(idx)
        # modify the source's target
        for index in node_info[source]["source"]:
            node_info[index]["target"].append(idx)
        node_info[target]["source"].append(idx)
        # modify target's source
        for index in node_info[target]["target"]:
            node_info[index]["source"].append(idx)
        # add all the possible link
        for ii in used_node:
            if ii not in node_info[idx]["source"] and ii != idx:
                possible_link.append((idx, ii))
            if ii not in node_info[idx]["target"] and ii != idx:
                possible_link.append((ii, idx))

        if link_candidate_selected in final_link:
            final_link.remove(link_candidate_selected)
        final_link.append((link_candidate_selected[0], idx))
        final_link.append((idx, link_candidate_selected[1]))

    final_link_matrix = np.zeros(test_case.shape)
    for value in final_link:
        final_link_matrix[value[0], value[1]] = 1

    return final_link, final_link_matrix

def main(params):
    dataset_file = params["dataset_file"]
    model_name = params["model_name"]
    split = params["split"]
    data_path = params["data_path"]
    seed = params["seed"]
    model_path = params["model_path"]
    np.random.seed(seed)
    random.seed(seed)

    with open(dataset_file) as f:
        data_info = json.load(f)


    tmp_data_info = {}
    for key, value in tqdm(data_info.items()):
        if value["split"] == split:
            tmp_data_info[key] = value

    data_info = tmp_data_info

    # load predictions
    eval_name = os.path.join(model_path, "runs", model_name, "eval", 'prediction_{}.pkl'.format(split))
    with open(eval_name, "rb") as f:
        predictions = pickle.load(f)

    # post pruning
    for key, value in tqdm(predictions.items()):
        # print("____________")
        ground_truth_graph = value["ground_truth_graph"]
        simplify_ground_truth_graph = simplify_graph(ground_truth_graph)
        prediction_matrix = predictions[key]["image_graph"][-1]
        final_link, final_link_matrix = post_pruning(prediction_matrix)
        final_link_matrix[0, 2:] = final_link_matrix[:, 2:].sum(0) > 0
        value["pruning_image_graph"] = final_link_matrix
        value["pruning_image_link"] = final_link

        prediction_matrix = predictions[key]["caption_graph"][-1]
        final_link, final_link_matrix = post_pruning(prediction_matrix)
        final_link_matrix[0, 2:] = final_link_matrix[:, 2:].sum(0) > 0
        value["pruning_caption_graph"] = final_link_matrix
        value["pruning_caption_link"] = final_link

    with open(os.path.join(model_path, "runs", model_name, "eval", 'post_pruning_{}.pkl'.format(split)), "wb") as f:
        pickle.dump(predictions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_file', default="/data/dataset.json", help='input file')
    parser.add_argument('--model_name', default="baseline", help='model name')
    parser.add_argument('--split', default="test", help='data split')
    parser.add_argument('--data_path', default="/data/images", help='data path')
    parser.add_argument('--seed', default=0, type=int, help='control the random seed')
    parser.add_argument('--model_path', default="/model/SGAN", help='model path')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)