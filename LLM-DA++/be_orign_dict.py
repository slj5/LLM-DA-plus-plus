import json
from utils.iotools import read_json
import argparse
from tqdm import tqdm



def decorate(args):
    annos = read_json(args.anno_path)
    aug_caption_sim_select_list = read_json(args.aug_caption_sim_select_list_path)
    dict_sort = {}
    annos_clean__sort = []
    for aug_caption_sim_select in tqdm(aug_caption_sim_select_list):
        key = aug_caption_sim_select['img_id']
        value = aug_caption_sim_select['gen_captions_list']
        existing_value = dict_sort.get(key, [])
        dict_sort[key] = existing_value + value


    for anno in tqdm(annos):
        img_id = anno['img_id']
        for key, value in dict_sort.items():
            if img_id == key:
                anno['captions_bt'] = value
        annos_clean__sort.append(anno)

    with open(args.save_path, 'w') as json_file:
        json.dump(annos_clean__sort, json_file)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", default='./train.json')
    parser.add_argument("--save_path", default='./train_aug.json')
    parser.add_argument("--aug_caption_sim_select_list_path",
                        default='./aug_caption_sim_select_list.json')
    args = parser.parse_args()

    decorate(args)
