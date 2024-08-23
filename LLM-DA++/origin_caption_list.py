
from utils.iotools import read_json
import json
from tqdm import tqdm

# regen_aug_caption_sim_list = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/regen_aug_caption_sim_list.json')
# aug_caption_list = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/aug_caption_list.json')
# aug_caption_sim_list = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/aug_caption_sim_list.json')
# aug_caption_sim_select_list = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/aug_caption_sim_select_list.json')
# test_annos  = read_json("/mnt/DataDrive/silijia/CUHK-PEDES/processed_data/test.json")
# aug_annos_0 = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/aug_annos_0.json')
# aug_annos_add_orign_even = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/aug_annos_add_orign_even.json')
# annos_aug_test = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/testtt/annos_aug_test.json')
# # anno_path = read_json('/mnt/DataDrive/silijia/ICFG-PEDES/ICFG-PEDES.json')
# aug_annos_add_orign_reverse = read_json('/mnt/DataDrive/silijia/CUHK-PEDES/vicuna/aug_annos_add_orign_reverse.json')
def _split_anno(anno_path: str):
    train_annos, test_annos, val_annos = [], [], []
    annos = read_json(anno_path)
    for anno in annos:
        if anno['split'] == 'train':
            train_annos.append(anno)
        elif anno['split'] == 'test':
            test_annos.append(anno)
        else:
            val_annos.append(anno)
    return train_annos, test_annos, val_annos

anno_path = '/mnt/DataDrive/silijia/ICFG-PEDES/ICFG-PEDES.json'
origin_caption_list_path = '/mnt/DataDrive/silijia/ICFG-PEDES/ICFG_PEDES_test_1000.json'
# test_path = '/mnt/DataDrive/silijia/CUHK-PEDES/processed_data/test_annos.json'


train_annos, test_annos, val_annos = _split_anno((anno_path))
ICFG_PEDES_test_1000 = []
for index, anno in tqdm(enumerate(test_annos)):
    ICFG_PEDES_test_1000.append(anno)
    # captions = anno['captions']
    # if len(captions) != 1:
    #     print(index)
    #     print(captions)
    # for caption in captions:
    #     origin_caption_list.append({'origin_caption': caption})
    if len(ICFG_PEDES_test_1000) == 1000:
        break

with open(origin_caption_list_path, 'w') as json_file:
    json.dump(ICFG_PEDES_test_1000, json_file)
#
# with open(test_path, 'w') as json_file:
#     json.dump(test_annos, json_file)