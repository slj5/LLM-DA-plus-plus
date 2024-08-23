import argparse
from sentence_transformers import SentenceTransformer, util
import json
from utils.iotools import read_json
import os
import os.path as op
import copy
import numpy as np
from fastchat.model import load_model, get_conversation_template, add_model_args
from tqdm import tqdm
import numpy as np


def select(args):
    aug_caption_sim_list = read_json(args.aug_caption_sim_list_path)
    aug_caption_sim_select_list = []
    regen_caption_index_list = []
    for caption_index, aug_caption_sim in tqdm(enumerate(aug_caption_sim_list)):
        gen_caption_list = aug_caption_sim['gen_captions_list']
        cos_sim_list = aug_caption_sim['cos_sim_list']
        select_gen_caption_list = []
        for cos_sim_index, cos_sim in enumerate(cos_sim_list):
            if cos_sim > args.threshold:
                select_gen_caption_list.append(gen_caption_list[cos_sim_index])
            if len(select_gen_caption_list) == args.select_times:
                break
        if len(gen_caption_list) < 5 and len(select_gen_caption_list) < args.select_times:
            regen_caption_index_list.append(caption_index)
        if len(gen_caption_list) >= 5 and len(select_gen_caption_list) < args.select_times:
            index_list = np.argpartition(cos_sim_list, -args.select_times)[-args.select_times:]
            index_list = index_list.tolist()
            for index in index_list:
                select_gen_caption_list.append(gen_caption_list[index])
        aug_caption_sim['select_gen_caption_list'] = select_gen_caption_list
        aug_caption_sim_select_list.append(aug_caption_sim)

    if regen_caption_index_list:
        print(len(regen_caption_index_list))
        print(aug_caption_sim_select_list[regen_caption_index_list[0]]['cos_sim_list'])

    with open(args.aug_caption_sim_select_list_path, 'w') as json_file:
        json.dump(aug_caption_sim_select_list, json_file)
    with open(args.regen_caption_index_list_path, 'w') as json_file:
        json.dump(regen_caption_index_list, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--aug_caption_sim_list_path", default='./aug_caption_sim_list.json')
    parser.add_argument("--aug_caption_sim_select_list_path", default='./aug_caption_sim_select_list.json')
    parser.add_argument("--regen_caption_index_list_path", default='./regen_caption_index_list.json')
    parser.add_argument("--threshold", type=float, default=0.6, help="the threshold of cosine similarity")
    args = parser.parse_args()
    select(args)


