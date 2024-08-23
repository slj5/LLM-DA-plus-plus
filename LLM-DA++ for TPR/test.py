import os
import random
import time
from pathlib import Path
import json
import torch

from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test
from misc.utils import parse_config, init_distributed_mode, set_seed_test, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model import clip_vitb
from options import get_args
from tqdm import tqdm

def run(config, run_times):
    print(config)

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)


    # model
    model = clip_vitb(config, num_classes)
    model.to(config.device)

    model, load_result = load_checkpoint(model, config)

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=True)


    eval_result = test(model.module, dataloader['test_loader'], 77, config.device, config, run_times)


    return  eval_result


if __name__ == '__main__':
    config_path = 'config/config_test.yaml'

    args = get_args()
    config = parse_config(config_path)
    n = config.experiment.n
    # Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)
    Path(config.experiment.save_matrix_path).mkdir(parents=True, exist_ok=True)
    Path(config.experiment.save_eval_result_path).mkdir(parents=True, exist_ok=True)

    save_eval_result_path = os.path.join(config.experiment.save_eval_result_path, 'eval_result_0.1.json')

    init_distributed_mode(config)
    best_rank1 = 0
    if config.experiment.deposit:
        eval_resulr_list = []
        for run_times in tqdm(range(n)):
            # set_seed(config)
            set_seed_test(run_times)
            eval_result = run(config, run_times)

            eval_resulr_list.append(eval_result)

            rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
            print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,                                                                                             top10=rank_10, mAP=map))
            print("run_times {} done!".format(run_times + 1))
        with open(save_eval_result_path, 'w') as json_file:
            json.dump(eval_resulr_list, json_file)
    elif config.experiment.withdraw:
        run_times = 0
        set_seed_test(run_times)
        eval_result = run(config, run_times)
        rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
        print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,
                                                                                          top10=rank_10, mAP=map))
    else:
        run_times = 0
        set_seed_test(run_times)
        eval_result = run(config, 0)
        rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
        print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,
                                                                                          top10=rank_10, mAP=map))







