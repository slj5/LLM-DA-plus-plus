import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args
import os.path as op
from typing import List
from utils.iotools import read_json
from tqdm import tqdm
import copy


@torch.inference_mode()
# call for vicuna
def call_model(args, model, tokenizer):
    msg = args.message
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(args.device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")
    return outputs

def vicuna_api(args):

    device = "cuda"
    aug_caption_list = []
    # prompt for vicuna
    Que = "Rewrite this image caption."

    # read the file for augmentation
    origin_caption_list = read_json(args.origin_caption_list_path)

    # 加载模型
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    # Sentence-by-sentence data augmentation experiments using LLM
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    for index, caption_dict in tqdm(enumerate(origin_caption_list)):
        args.message = caption_dict['origin_caption'] + ' ' + Que
        outputs = call_model(args, model, tokenizer)
        caption_dict['gen_captions_list'].append(outputs)
        aug_caption_list.append(caption_dict)
        # Save 100  original sentence at a time
        if not (index + 1) % 100:
            with open(args.aug_caption_list_path, 'w') as json_file:
                json.dump(aug_caption_list, json_file)

    # save the augmented file
    with open(args.aug_caption_list_path, 'w') as json_file:
        json.dump(aug_caption_list, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str)
    parser.add_argument("--model_path", default='./vicuna/model', help='the path of storing vicuna')
    parser.add_argument("--origin_caption_list_path", default='./origin_caption_list.json')
    parser.add_argument("--aug_caption_list_path", default='./aug_caption_list.json', help='the save dir of augmented files')
    args = parser.parse_args()
    vicuna_api(args)

