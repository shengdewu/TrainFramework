import torch.distributed as dist
import torch
import torch.multiprocessing as mp
from engine.config.parser import default_argument_parser
from engine.config import get_cfg
import engine.comm as comm
from fvcore.common.config import CfgNode
from fvcore.common.file_io import PathManager
from iopath.common.file_io import g_pathmgr
import yaml
from engine.log.logger import setup_logger
import logging
import os
import engine.collect_env as collect_env
from engine.trainer.build import build_trainer
import tempfile
import shutil
import sys
import types
from importlib import import_module


BASE_KEY = '_base_'
LR_SCHEDULER = 'LR_SCHEDULER'
OPTIMIZER = 'OPTIMIZER'
TYPE = 'TYPE'
PARAMS = 'PARAMS'


def upper(k):
    new_k = k.upper()
    if new_k.lower() != k:
        new_k = k
    return new_k


class BaseScheduler:
    def __init__(self):
        return

    def lunch_func(self, cfg, args):
        """
        create trainer, load traienr, start train

        Args:
            cfg :ymal
            args :cmd
        """
        trainer = build_trainer(cfg)
        trainer.resume_or_load(args.resume)
        trainer.loop()
        return

    def main_func(self, args, local_rank=None, global_rank=0, world_size=1, is_distributed=False):
        cfg = self.setup(args)

        cfg.defrost()
        if is_distributed:
            cfg.TRAINER.PARADIGM.TYPE = 'DDP'  # -1 :normal, 0:parallel, 1 :distributed
            cfg.TRAINER.PARADIGM.GPU_ID = local_rank
            cfg.TRAINER.PARADIGM.GLOBAL_RANK = global_rank
            cfg.TRAINER.PARADIGM.WORLD_SIZE = world_size
            cfg.TRAINER.PARADIGM.NUM_PER_GPUS = args.num_gpus
            cfg.TRAINER.DEVICE = 'cuda'
        else:
            if torch.cuda.is_available() and args.num_gpus >= 2:
                cfg.TRAINER.PARADIGM.TYPE = 'DP'  # -1 :normal, 0:parallel, 1 :distributed
                cfg.TRAINER.PARADIGM.GPU_ID = None
                cfg.TRAINER.PARADIGM.GLOBAL_RANK = 0
                cfg.TRAINER.PARADIGM.WORLD_SIZE = 1
                cfg.TRAINER.PARADIGM.NUM_PER_GPUS = args.num_gpus
                cfg.TRAINER.DEVICE = 'cuda'
            else:
                cfg.TRAINER.PARADIGM.TYPE = 'NORMAL'  # -1 :normal, 0:parallel, 1 :distributed
                cfg.TRAINER.PARADIGM.GPU_ID = None
                cfg.TRAINER.PARADIGM.GLOBAL_RANK = 0
                cfg.TRAINER.PARADIGM.WORLD_SIZE = 1
                cfg.TRAINER.PARADIGM.NUM_PER_GPUS = 1
                if cfg.TRAINER.DEVICE == '' or cfg.TRAINER.DEVICE is None:
                    cfg.TRAINER.DEVICE = 'cuda' if torch.cuda.is_available() and args.num_gpus > 0 else 'cpu'
                if not torch.cuda.is_available():
                    cfg.TRAINER.DEVICE = 'cpu'

        cfg.freeze()

        rank = comm.get_rank()  # rank of current node,[0, nodes - 1] and nodes : total number of nodes
        logger = setup_logger(cfg.OUTPUT_DIR, rank, name=cfg.OUTPUT_LOG_NAME)
        if comm.is_main_process() and cfg.OUTPUT_DIR:
            logger.info("Environment info:\n {}".format(collect_env.collect_env_info()))
            logger.info("is distribute {} Running with full config:\n{}".format(is_distributed, cfg.dump()))

            path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")

            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())

        self.lunch_func(cfg, args)
        return

    def distributed(self, local_rank, loop_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args):
        assert torch.cuda.is_available(), "cuda is not available. Please check your installation."

        global_rank = machine_rank * num_gpus_per_machine + local_rank

        torch.distributed.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=global_rank)

        # Setup the local process group (which contains ranks within the same machine)
        assert comm._LOCAL_PROCESS_GROUP is None
        num_machines = world_size // num_gpus_per_machine
        for i in range(num_machines):
            ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
            pg = dist.new_group(ranks_on_i)
            if i == machine_rank:
                comm._LOCAL_PROCESS_GROUP = pg

        assert num_gpus_per_machine <= torch.cuda.device_count()

        torch.cuda.set_device(local_rank)

        comm.synchronize()

        loop_func(args, local_rank, global_rank, world_size, True)

        return

    @staticmethod
    def _find_free_port():
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Binding to port 0 will cause the OS to find an available port for us
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        # NOTE: there is still a chance the port could be taken by other processes.
        return port

    def align_struct(self, default_cfg, user_cfg):
        lr_config = default_cfg.SOLVER.GENERATOR.LR_SCHEDULER
        user_lr_config = user_cfg['SOLVER']['GENERATOR']['LR_SCHEDULER']
        if lr_config.TYPE != user_lr_config['TYPE']:
            default_cfg.SOLVER.GENERATOR.LR_SCHEDULER.PARAMS = CfgNode(new_allowed=True)

        op_config = default_cfg.SOLVER.GENERATOR.OPTIMIZER
        user_op_config = user_cfg['SOLVER']['GENERATOR']['OPTIMIZER']
        if op_config.TYPE != user_op_config['TYPE']:
            default_cfg.SOLVER.GENERATOR.OPTIMIZER.PARAMS = CfgNode(new_allowed=True)

        # discriminator
        user_dicriminator_config = user_cfg['SOLVER'].get('DISCRIMINATOR', None)
        if user_dicriminator_config is not None:
            if isinstance(user_dicriminator_config, list):
                default_cfg.SOLVER.DISCRIMINATOR = user_cfg['SOLVER']['DISCRIMINATOR']
            else:
                lr_config = default_cfg.SOLVER.DISCRIMINATOR.get('LR_SCHEDULER', None)
                if lr_config is None:
                    default_cfg.SOLVER.DISCRIMINATOR.LR_SCHEDULER = CfgNode(new_allowed=True)
                    default_cfg.SOLVER.DISCRIMINATOR.LR_SCHEDULER.PARAMS = CfgNode(new_allowed=True)
                else:
                    user_lr_config = user_dicriminator_config['LR_SCHEDULER']
                    if lr_config is None or lr_config.TYPE != user_lr_config['TYPE']:
                        default_cfg.SOLVER.DISCRIMINATOR.LR_SCHEDULER.PARAMS = CfgNode(new_allowed=True)

                op_config = default_cfg.SOLVER.DISCRIMINATOR.get('OPTIMIZER', None)
                if op_config is None:
                    default_cfg.SOLVER.DISCRIMINATOR.OPTIMIZER = CfgNode(new_allowed=True)
                    default_cfg.SOLVER.DISCRIMINATOR.OPTIMIZER.PARAMS = CfgNode(new_allowed=True)
                else:
                    user_op_config = user_dicriminator_config['OPTIMIZER']
                    if op_config is None or op_config.TYPE != user_op_config['TYPE']:
                        default_cfg.SOLVER.DISCRIMINATOR.OPTIMIZER.PARAMS = CfgNode(new_allowed=True)

        user_dicriminator_config = user_cfg['TRAINER']['MODEL'].get('DISCRIMINATOR', None)
        if user_dicriminator_config is not None and isinstance(user_dicriminator_config, list):
            default_cfg.TRAINER.MODEL.DISCRIMINATOR = user_cfg['TRAINER']['MODEL']['DISCRIMINATOR']

        return default_cfg

    @staticmethod
    def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
        if not os.path.isfile(filename):
            raise FileNotFoundError(msg_tmpl.format(filename))

    def dict2cfg(self, config, is_cfg=True):
        uppre_config = dict()
        for k, v in config.items():
            if isinstance(v, dict):
                uppre_config[upper(k)] = self.dict2cfg(v, is_cfg)
            elif isinstance(v, list):
                if isinstance(v[0], dict):
                    uppre_config[upper(k)] = list()
                    for vv in v:
                        uppre_config[upper(k)].append(self.dict2cfg(vv, is_cfg=False))
                else:
                    uppre_config[upper(k)] = v.copy()
            else:
                uppre_config[upper(k)] = v
        if is_cfg:
            uppre_config = CfgNode(init_dict=uppre_config)
        return uppre_config

    def merge_base(self, base_cfg: dict, cfg: dict):
        for key, value in base_cfg.items():
            if key.lower() == LR_SCHEDULER.lower() or key.lower() == OPTIMIZER.lower():
                if key not in cfg.keys():
                    cfg[key] = value
                else:
                    if value[TYPE.lower()] != cfg[key][TYPE.lower()]:
                        print(f'use the cfg to replace the value of base in {key}')
                        continue
                    else:
                        if isinstance(value, dict):
                            self.merge_base(value, cfg[key])
                        else:
                            print(f'use the cfg to replace the value of base in {key}')
            else:
                if key not in cfg.keys():
                    cfg[key] = value
                else:
                    if isinstance(value, dict):
                        self.merge_base(value, cfg[key])
                    else:
                        print(f'use the cfg to replace the value of base in {key}')
        return

    def py2cfg(self, file_name, to_cfg=True):
        file_name = os.path.abspath((os.path.expanduser(file_name)))
        self.check_file_exist(file_name)
        cfg = get_cfg()
        keys = [key.lower() for key in cfg.keys()]
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix='.py')
            temp_config_name = os.path.basename(temp_config_file.name)
            shutil.copyfile(file_name, temp_config_file.name)
            temp_module_name = os.path.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
                   and not isinstance(value, types.ModuleType)
                   and not isinstance(value, types.FunctionType)
                   and name.lower() in keys
            }

            base_dict = dict()
            for name, value in mod.__dict__.items():
                if name.lower() != BASE_KEY:
                    continue
                base_filename = value if isinstance(value, list) else [value]
                for base in base_filename:
                    base_dict.update(self.py2cfg(f'{os.path.dirname(file_name)}/{base}', to_cfg=False))

            del sys.modules[temp_module_name]
            temp_config_file.close()

        if len(base_dict) > 0:
            self.merge_base(base_dict, cfg_dict)

        return self.dict2cfg(cfg_dict) if to_cfg else cfg_dict

    def setup(self, args):
        cfg = get_cfg()
        if args.config_file.endswith('.py'):
            user_cfg = self.py2cfg(args.config_file)
        else:
            with g_pathmgr.open(args.config_file, "r") as f:
                user_cfg = self.dict2cfg(yaml.safe_load(f))

        cfg = self.align_struct(cfg, user_cfg)
        cfg.merge_from_other_cfg(user_cfg)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

        if comm.is_main_process() and cfg.OUTPUT_DIR:
            PathManager.mkdirs(cfg.OUTPUT_DIR)
        return cfg

    def schedule(self):
        args = default_argument_parser().parse_args()

        num_gpus_per_machine = args.num_gpus
        num_machines = args.num_machines
        machine_rank = args.machine_rank  # current machine node no.
        dist_url = args.dist_url

        world_size = num_machines * num_gpus_per_machine
        if args.distribute and world_size > 1:
            # https://github.com/pytorch/pytorch/pull/14391
            # TODO prctl in spawned processes

            if dist_url == "auto":
                assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
                port = self._find_free_port()
                dist_url = f"tcp://127.0.0.1:{port}"
            if num_machines > 1 and dist_url.startswith("file://"):
                logger = logging.getLogger(__name__)
                logger.warning(
                    "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
                )

            mp.spawn(self.distributed, nprocs=num_gpus_per_machine, args=(self.main_func,
                                                                          world_size,
                                                                          num_gpus_per_machine,
                                                                          machine_rank,
                                                                          dist_url,
                                                                          args))
        else:
            self.main_func(args, is_distributed=False)

        return
