import argparse
import json
import os
import torch

from fastchat.model import load_model, get_conversation_template, add_model_args

import os.path as op
from typing import List
from utils.iotools import read_json
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from gen import vicuna_api
from recom_sim import count_cossim
from filter import select
from utils.iotools import read_json

def judge(args):
    select(args)
    regen_caption_index_list = read_json(args.regen_caption_index_list_path)
    while regen_caption_index_list:
        vicuna_api(args)
        count_cossim(args)
        select(args)
        regen_caption_index_list = read_json(args.regen_caption_index_list_path)
        if not regen_caption_index_list:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str)
    parser.add_argument("--model_path", default='./model')
    parser.add_argument("--aug_caption_sim_list_path", default='./aug_caption_sim_list.json')
    parser.add_argument("--regen_caption_index_list_path", default='./regen_caption_index_list.json')
    parser.add_argument("--aug_caption_sim_select_list_path", default='./aug_caption_sim_select_list.json')
    parser.add_argument("--threshold", type=float, default=0.8, help="the threshold of cosine similarity")
    args = parser.parse_args()

    judge(args)
