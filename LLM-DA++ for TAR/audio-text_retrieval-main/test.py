import platform
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pprint import PrettyPrinter
from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from models.ASE_model import ASE
from data_handling.DataLoader import get_dataloader
from tools.config_loader import get_config
import argparse


def validate(data_loader, model, device, config):

    # val_logger = logger.bind(indent=1)
    model.eval()
    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):

            if config.test_sample:
                audios, captions, captions_aug, audio_ids, indexs = batch_data
                # move data to GPU
                audios = audios.to(device)
                # print(audios.device)
                # print(audio_ids.device)
                audio_embeds, caption_embeds = model(audios, captions, captions_aug)
            else:
                audios, captions, audio_ids, indexs = batch_data
                # move data to GPU
                audios = audios.to(device)
                audio_embeds, caption_embeds = model(audios, captions, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        # evaluate text to audio retrieval
        r1, r5, r10, r50, medr, meanr = t2a(audio_embs, cap_embs)

        print('Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1, r5, r10, r50, medr, meanr))
        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t(audio_embs, cap_embs)

        print('Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a))
        # return r1, r5, r10, r50, medr, meanr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')
    args = parser.parse_args()
    config = get_config(args.config)
    model_output_dir = ' '
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())

    test_loader = get_dataloader('test', config, False, config.test_sample)

    model = ASE(config)
    model = model.to(device)
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')

    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    validate(test_loader, model, device, config)



