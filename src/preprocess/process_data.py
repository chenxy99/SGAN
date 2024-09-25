import os
import json
from tqdm import tqdm
import numpy as np
import random
from collections import OrderedDict
import argparse
import html

def main(params):
    seed = params["seed"]
    np.random.seed(seed)
    random.seed(seed)

    input_file = params["input_file"]
    output_file = params["output_file"]

    with open(input_file) as f:
        data_info = json.load(f)

    for key, value in tqdm(data_info.items()):
        if isinstance(value["step_to_dependency_index_result"], list):
            pass
        else:
            value["step_to_dependency_index_result"] = json.loads(value["step_to_dependency_index_result"])

    # change the html -> string
    for key, value in tqdm(data_info.items()):
        step_num = len(value["step_to_object_selected_result"])
        for index in range(step_num):
            curr_obj = value["step_to_object_selected_result"][index]
            grounding_obj = curr_obj[0]
            grounding_obj = [html.unescape(_) for _ in grounding_obj]

            non_grounding_obj = curr_obj[1]
            non_grounding_obj = [html.unescape(_) for _ in non_grounding_obj]

            value["step_to_object_selected_result"][index][0] = grounding_obj
            value["step_to_object_selected_result"][index][1] = non_grounding_obj

    # change the html -> string
    for key, value in tqdm(data_info.items()):
        step_num = len(value["step_to_object_bbox_result"])
        for index in range(step_num):
            curr_obj = value["step_to_object_bbox_result"][index]
            grounding_obj = curr_obj[0]
            for _ in grounding_obj:
                for val in _:
                    val['label'] = html.unescape(val['label'])

            non_grounding_obj = curr_obj[2]
            for _ in non_grounding_obj:
                for val in _:
                    val['label'] = html.unescape(val['label'])


    with open(output_file, "w") as f:
        json.dump(data_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default="../data/wikihow/final_result_w_split_20028_raw_fixed_order.json", help='input file')
    parser.add_argument('--output_file', default="../data/wikihow/final_result_w_split_20028.json", help='output file')

    parser.add_argument('--seed', default=0, type=int, help='control the random seed')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)