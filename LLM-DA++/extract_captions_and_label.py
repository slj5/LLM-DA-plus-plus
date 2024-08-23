from utils.iotools import read_json
import json
from tqdm import tqdm

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

anno_path = './CUHK-PEDES/reid_raw.json'
train_annos, test_annos, val_annos = _split_anno((anno_path))
origin_caption_list_path = './origin_caption_list.json.json'
origin_caption_list = []
caption_id = 0
for index, anno in tqdm(enumerate(train_annos)):
    captions = anno['captions']
    person_id = anno['id']
    for caption_index, caption in enumerate(captions):
        dict = {}
        dict['origin_caption'] = caption
        dict['caption_id'] = caption_id
        dict['id'] = person_id
        dict['img_id'] = index
        dict['gen_captions_list'] = []
        caption_id += 1
        origin_caption_list.append(dict)

print(caption_id)
print(len(origin_caption_list))
with open(origin_caption_list_path, 'w') as json_file:
    json.dump(origin_caption_list, json_file)
