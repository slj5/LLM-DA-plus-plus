import json
from utils.iotools import read_json
import argparse
from tqdm import tqdm
import copy


def decorate(args):
    # 需要考虑原始数据集中文本的数目，同时还要考虑倍数，最后生成数据集的数目
    aug_caption_sim_select_list = read_json(args.aug_caption_sim_select_list_path)
    annos = read_json(args.anno_path)
    length = len(annos[0]['captions'])
    select_times = args.select_times
    select_caption_list = []
    aug_annos = {}
    for aug_caption_sim_select in aug_caption_sim_select_list:
        select_caption_list.append(aug_caption_sim_select['select_gen_caption_list'])
    # if length == 2:
    for times in range(select_times):
        list_name = f"aug_annos_0.8_{times}"
        aug_annos[list_name] = []
        current_text_index = 0
        for index, dictionary in tqdm(enumerate(annos)):
            # if (len(dictionary['captions'])) != 2:
            #     print("inex:", index, "length:", len(dictionary['captions']))
            len_captions = len(dictionary['captions'])
            dictionary_aug = copy.deepcopy(dictionary)
            dictionary_aug['captions'] = []
            for len_caption in range(len_captions):
                dictionary_aug['captions'].append(select_caption_list[current_text_index + len_caption][times])
            current_text_index = current_text_index + len_captions
            # for len_caption in range(len_captions):
            #     dictionary_aug['captions'].append(select_caption_list[2 * index + len_caption][times])
            aug_annos[list_name].append(dictionary_aug)

    for key, value in aug_annos.items():
        file_name = f"{args.save_path}{key}.json"
        with open(file_name, 'w') as json_file:
            json.dump(value, json_file)
    # else:
    #     for times in range(select_times):
    #         list_name = f"aug_annos_{times}"
    #         aug_annos[list_name] = []
    #         for index, dictionary in tqdm(enumerate(annos)):
    #             len_captions = len(dictionary['captions'])
    #             dictionary_aug = copy.deepcopy(dictionary)
    #             dictionary_aug['captions'] = []
    #             for len_caption in range(len_captions):
    #                 dictionary_aug['captions'].append(select_caption_list[index + len_caption][times])
    #             aug_annos[list_name].append(dictionary_aug)

        # for key, value in aug_annos.items():
        #     file_name = f"{args.save_path}{key}.json"
        #     with open(file_name, 'w') as json_file:
        #         json.dump(value, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", default='/mnt/DataDrive/silijia/ICFG-PEDES/contant/ICFG_PEDES_test_1000.json')
    parser.add_argument("--save_path", default='/mnt/DataDrive/silijia/ICFG-PEDES/contant_test/', help='the save path of final augment file')
    # parser.add_argument("--regen_caption_index_list_path",
    #                     default='/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/regen_caption_index_list.json')
    parser.add_argument("--aug_caption_sim_select_list_path",
                        default='/mnt/DataDrive/silijia/ICFG-PEDES/vicuna_aug/aug_caption_sim_select_list.json')
    parser.add_argument("--select_times", type=int, default=1, help="select times of text")
    args = parser.parse_args()

    decorate(args)
