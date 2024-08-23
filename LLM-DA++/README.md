# LLM-DA code explaination
Please run the code strictly in the following order

## extract_captions_and_label.py: 
### function： extract the original captions and label them
### input： original path of datdaset such as: reid_raw.json
### output：origin_caption_list.json


## gen.py:
### function： call for vicuna for text augmentation
### input： origin_caption_list.json
### output： aug_caption_list.json


## comp_sim.py：
### function： compute the similarity of original text and augmented text. The range of values is [-1,1].
### input： aug_caption_list.json
### output： aug_caption_sim_list.json (This is an incremental file)

## filter.py:
### function： According to threshold to select the augmented texts
### input： aug_caption_sim_list.json
### output： aug_caption_sim_select_list.json（compliant augmented text）、regen_caption_index_list.json（indexing of non-conforming augmented text）

## judge.py：
### function：For the text that needs to be regenerated, repeat the operations of regenerating, calculating cosine similarity, filtering, etc., until the conditions are satisfied
### input：aug_caption_sim_list.json、regen_caption_index_list.json、
### output： aug_caption_sim_select_list.json

## be_orign_dict.py:
### function： Integrate augmented text into the data format needed for the TPR task as required
### input： aug_caption_sim_select_list.json、train.json
### output： train_aug.json


