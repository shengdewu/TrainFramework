import torch.distributed as tdistributed
import os
import torch
import logging
import torch.multiprocessing as mp
import tframework.log.Log
from abc import ABC


class BaseSchedule(ABC):
    def __init__(self):
        return

    def is_main_process(self):
        if not tdistributed.is_available():
            return True
        if not tdistributed.is_initialized():
            return True
        else:
            return tdistributed.get_rank() == 0

    def create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def loop(self, model, train_loader, network_config, valid_dataset):
        pass

    def create_distribute_model(self, config, gpu_id, rank, dataset):
        return None, None

    def create_parallel_model(self, config, dataset):
        return None, None

    def create_dataset(self, config):
        return None, None

    def non_distribute(self, network_config, dataset, valid_dataset):
        if network_config['gpu'] <= 0 or torch.cuda.device_count() <= 1:
            network_config['data_parallel'] = False

        model, train_data_loader = self.create_parallel_model(network_config, dataset)

        tframework.log.Log.Log.init_log('detect', network_config['out_path'] + '/log')
        logging.info('start train, param {}'.format(network_config))

        self.loop(model, train_data_loader, network_config, valid_dataset)
        return

    def distributed(self, gpu_id, network_config, dataset, valid_dataset):
        rank = gpu_id + network_config['node_rank'] * network_config['world_size'] # single machine
        torch.distributed.init_process_group('nccl', init_method='env://', world_size=network_config['world_size'], rank=rank)
        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)

        model, train_data_loader = self.create_distribute_model(network_config, gpu_id, rank, dataset)

        if self.is_main_process():
            tframework.log.Log.Log.init_log('detect', network_config['out_path'] + '/log')
            logging.info('start train, param {}'.format(network_config))

        self.loop(model, train_data_loader, network_config, valid_dataset)
        return

    def schedule(self, network_config):

        network_config['save_image_path'] = self.create_path(os.path.join(network_config['out_path'], 'out_image'))
        network_config['save_model_path'] = self.create_path(os.path.join(network_config['out_path'], 'model_path'))
        network_config['save_summary_path'] = self.create_path(os.path.join(network_config['out_path'], 'model_summary'))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if network_config['gpu'] <= 0:
            device = 'cpu'

        network_config['device'] = device

        train_dataset, valid_dataset = self.create_dataset(network_config)

        if not network_config['distributed'] or device == 'cpu':
            self.non_distribute(network_config, train_dataset, valid_dataset)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('the machine have not gpu')

            total_gpus = torch.cuda.device_count()

            use_gpu_num = network_config['gpu'] if network_config['gpu'] < total_gpus else total_gpus

            network_config['nodes'] = 1  # total number of nodes
            network_config['world_size'] = use_gpu_num * network_config['nodes']  # use_gpu_num must be equal between all nodes
            network_config['node_rank'] = 0  # rank of current node,[0, nodes - 1]
            os.environ['MASTER_ADDR'] = 'localhost'  # the addr of master node
            os.environ['MASTER_PORT'] = '12355'  # the port of master node

            mp.spawn(self.distributed, nprocs=use_gpu_num, args=(network_config, train_dataset, valid_dataset))
        return

