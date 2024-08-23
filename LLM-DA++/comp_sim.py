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

# call for SentenceTransformer
model_cossim = SentenceTransformer('./paraphrase-MiniLM-L12-v2')


def count_cossim(anno_path):
    # read file
    aug_caption_list = read_json(anno_path)
    aug_caption_sim_list = []
    for caption_dict in tqdm(aug_caption_list):
        origin_caption = caption_dict['origin_caption']
        gen_caption_list = caption_dict['gen_captions_list']
        caption_dict['cos_sim_list'] = []
        emb1 = model_cossim.encode(origin_caption)
        for gen_caption in gen_caption_list:
            emb2 = model_cossim.encode(gen_caption)
            cos_sim = util.cos_sim(emb1, emb2)  # output tensors

            # convert tensor to python list
            numpy_cos_sim = cos_sim.cpu().numpy().astype(np.float64)
            cos_sim_list = numpy_cos_sim.flatten()
            cos_sim = np.ndarray.tolist(cos_sim_list)[0]
            caption_dict['cos_sim_list'].append(cos_sim)

        aug_caption_sim_list.append(caption_dict)

    return aug_caption_sim_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--anno_path", default='./aug_caption_list.json')
    parser.add_argument("--save_path", default='./aug_caption_sim_list.json', help='the save path of file that have similarity')
    args = parser.parse_args()

    anno_aug_sim_list = count_cossim(args.anno_path)
    with open(args.save_path, 'w') as json_file:
        json.dump(anno_aug_sim_list, json_file)

