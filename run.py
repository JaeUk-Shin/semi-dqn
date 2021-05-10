import json
import argparse
from multiprocessing import Process, set_start_method
import os
from torch.cuda import device_count
from semidqn_prioritized import run_prioritized


if __name__ == '__main__':
    # gpu_count = device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_pth', required=True, type=str)
    args = parser.parse_args()
    cfg_pth = os.path.abspath(args.cfg_pth)
    pth = cfg_pth.replace(os.getcwd() + '/configs/', '') + '/'

    set_start_method('spawn')

    device: int = 0
    for file_name in os.listdir(cfg_pth):
        if file_name.endswith('.json'):
            f = open(cfg_pth + '/' + file_name)
            params = json.load(f)
            # params['device'] = 'cuda:{}'.format(device)
            log_pth = './log/' + pth + file_name.replace('.json', '') + '/'
            params['pth'] = log_pth

            # TODO : directory for log

            p = Process(target=run_prioritized, kwargs=params)
            p.start()
            # device = (device + 1) % gpu_count
