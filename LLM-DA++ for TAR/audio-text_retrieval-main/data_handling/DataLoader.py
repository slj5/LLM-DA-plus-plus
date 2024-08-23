
import torch
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class AudioCaptionDataset(Dataset):

    def __init__(self, dataset='AudioCaps', split='train', train_sample=False, test_sample=True):
        """
        load audio clip's waveform and corresponding caption
        Args:
            dataset: 'AudioCaps', 'Clotho
            split: 'train', 'val', 'test'
        """
        super(AudioCaptionDataset, self).__init__()
        self.dataset = dataset
        self.split = split
        self.h5_path = f'data/{dataset}/hdf5s/{split}/{split}.h5'
        self.train_sample = train_sample
        self.test_sample = test_sample
        if self.train_sample and split == 'train':
            self.h5_path = f'data/{dataset}/hdf5s/{split}_aug_0.4/{split}.h5'
        if self.test_sample and split == 'test':
            self.h5_path = f'data/{dataset}/hdf5s/{split}_aug_0.8/{split}.h5'

        if dataset == 'AudioCaps' and split == 'train':
            self.is_train = True
            self.num_captions_per_audio = 1
            with h5py.File(self.h5_path, 'r') as hf:
                self.audio_keys = [audio_name.decode() for audio_name in hf['audio_name'][:]]
                # audio_names: [str]
                self.captions = [caption.decode() for caption in hf['caption'][:]]
        else:
            self.is_train = False
            self.num_captions_per_audio = 5
            with h5py.File(self.h5_path, 'r') as hf:
                self.audio_keys = [audio_name.decode() for audio_name in hf['audio_name'][:]]
                self.captions = [caption for caption in hf['caption'][:]]
                if self.train_sample and self.split == 'train':
                    self.captions_aug = [caption for caption in hf['caption_aug'][:]]
                if self.test_sample and self.split == 'test':
                    self.captions_aug = [caption for caption in hf['caption_aug'][:]]
                if dataset == 'Clotho':
                    self.audio_lengths = [length for length in hf['audio_length'][:]]
                # [cap_1, cap_2, ..., cap_5]

    def __len__(self):
        return len(self.audio_keys) * self.num_captions_per_audio

    def __getitem__(self, index):

        audio_idx = index // self.num_captions_per_audio
        audio_name = self.audio_keys[audio_idx]
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = hf['waveform'][audio_idx]

        if self.dataset == 'AudioCaps' and self.is_train:
            caption = self.captions[audio_idx]
        else:
            captions = self.captions[audio_idx]
            cap_idx = index % self.num_captions_per_audio
            caption = captions[cap_idx].decode()
            if self.train_sample and self.split == 'train':
                captions_aug = self.captions_aug[audio_idx]
                cap_idx = index % self.num_captions_per_audio
                caption_aug = captions_aug[cap_idx].decode()

            if self.test_sample and self.split == 'test':
                captions_aug = self.captions_aug[audio_idx]
                cap_idx = index % self.num_captions_per_audio
                caption_aug = captions_aug[cap_idx].decode()

        if self.dataset == 'Clotho' and self.train_sample and self.split == 'train':
            length = self.audio_lengths[audio_idx]
            return waveform, caption, caption_aug, audio_idx, length, index
        if self.dataset == 'Clotho' and self.test_sample and self.split == 'test':
            length = self.audio_lengths[audio_idx]
            return waveform, caption, caption_aug, audio_idx, length, index
        if self.dataset == 'Clotho':
            length = self.audio_lengths[audio_idx]
            return waveform, caption, audio_idx, length, index
        else:
            return waveform, caption, audio_idx, len(waveform), index


def collate_fn_train_aug(batch_data):
    """

    Args:
        batch_data:

    Returns:

    """

    # max_audio_length = max([i[3] for i in batch_data])
    max_audio_length = max([i[4] for i in batch_data])
    wav_tensor = []
    for waveform, _, _, _, _, _ in batch_data:
        if max_audio_length > waveform.shape[0]:
            padding = torch.zeros(max_audio_length - waveform.shape[0]).float()
            temp_audio = torch.cat([torch.from_numpy(waveform).float(), padding])
        else:
            temp_audio = torch.from_numpy(waveform[:max_audio_length]).float()
        wav_tensor.append(temp_audio.unsqueeze_(0))

    wavs_tensor = torch.cat(wav_tensor)
    captions = [i[1] for i in batch_data]
    captions_aug = [i[2] for i in batch_data]
    # audio_ids = torch.Tensor([i[2] for i in batch_data])
    audio_ids = torch.Tensor([i[3] for i in batch_data])
    # indexs = np.array([i[4] for i in batch_data])
    indexs = np.array([i[5] for i in batch_data])

    return wavs_tensor, captions, captions_aug, audio_ids, indexs

def collate_fn_else(batch_data):
    """

    Args:
        batch_data:

    Returns:

    """

    max_audio_length = max([i[3] for i in batch_data])
    wav_tensor = []
    for waveform, _, _, _, _ in batch_data:
        if max_audio_length > waveform.shape[0]:
            padding = torch.zeros(max_audio_length - waveform.shape[0]).float()
            temp_audio = torch.cat([torch.from_numpy(waveform).float(), padding])
        else:
            temp_audio = torch.from_numpy(waveform[:max_audio_length]).float()
        wav_tensor.append(temp_audio.unsqueeze_(0))

    wavs_tensor = torch.cat(wav_tensor)
    captions = [i[1] for i in batch_data]
    audio_ids = torch.Tensor([i[2] for i in batch_data])
    indexs = np.array([i[4] for i in batch_data])

    return wavs_tensor, captions, audio_ids, indexs


def get_dataloader(split, config, train_sample, test_sample):
    dataset = AudioCaptionDataset(config.dataset, split, train_sample, test_sample)
    if split == 'train' and train_sample:
        shuffle = True
        drop_last = True
        collate_fn = collate_fn_train_aug

    elif split == 'test' and test_sample:
        shuffle = False
        drop_last = False
        collate_fn = collate_fn_train_aug
    else:
        if split == 'train':
            shuffle = True
            drop_last = True
            collate_fn = collate_fn_else
        else:
            shuffle = False
            drop_last = False
            collate_fn = collate_fn_else


    return DataLoader(dataset=dataset,
                      batch_size=config.data.batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=config.data.num_workers,
                      collate_fn=collate_fn)


