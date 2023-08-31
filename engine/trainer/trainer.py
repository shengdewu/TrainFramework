import logging
from engine.model.base_model import BaseModel
import engine.checkpoint.checkpoint_manager as engine_checkpoint_manager
import engine.comm as comm
import engine.data.data_loader as engine_data_loader
from engine.model.build import build_model
from engine.data.build import build_dataset
from torch.utils.data import Dataset
import torch
from .build import BUILD_TRAINER_REGISTRY


@BUILD_TRAINER_REGISTRY.register()
class BaseTrainer:
    """
    first:
        1. create model in __init__
        2. create dataloader in __init__
        3. create checkpointer in __init__
        4. init log use setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__) in engine.log.logger
        4. training in loop
        5. resume model in resume_or_load
    then:
        called by subclass of BaseScheduler in engine/schedule.py that must be implement abc lunch_func
    eg:
        def lunch_func(self, cfg, args):
            trainer = BaseTrainer(cfg)
            trainer.resume_or_load(args.resume)
            trainer.loop()
        return
    ps:
        model may be by registered use Registry (from fvcore.common.registry import)
    """

    def __init__(self, cfg):
        self.default_log_name = cfg.OUTPUT_LOG_NAME

        self.model = self.create_model(cfg)
        self.model.enable_train()
        self.model.enable_distribute(cfg)

        model_name = self.model.g_model.__class__.__name__
        if isinstance(self.model.g_model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model_name = self.model.g_model.module.__class__.__name__

        self.checkpoint = engine_checkpoint_manager.CheckPointerManager(max_iter=cfg.SOLVER.MAX_ITER,
                                                                        save_dir=cfg.OUTPUT_DIR,
                                                                        check_period=cfg.SOLVER.CHECKPOINT_PERIOD,
                                                                        max_keep=cfg.SOLVER.MAX_KEEP,
                                                                        file_prefix=model_name,
                                                                        save_to_disk=comm.is_main_process())
        self.collate_train_fn = None
        self.collate_valid_fn = None
        self.set_collate_fn(cfg)

        train_dataset, valid_dataset = self.create_dataset(cfg)
        train_data_loader, test_data_loader = self.create_dataloader(cfg, train_dataset, valid_dataset)

        self.start_iter = 0
        self.model_path = cfg.TRAINER.WEIGHTS
        self.device = cfg.TRAINER.DEVICE
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.output = cfg.OUTPUT_DIR
        self.enable_epoch_method = cfg.TRAINER.ENABLE_EPOCH_METHOD

        self.train_data_loader = iter(train_data_loader) if not self.enable_epoch_method else train_data_loader
        self.test_data_loader = test_data_loader

        total_data_per_epoch = len(train_dataset) / cfg.SOLVER.TRAIN_PER_BATCH

        format_string = 'create train dataset {}\n'.format(train_dataset)
        format_string += 'create valid dataset {}\n'.format(valid_dataset)
        format_string += 'there are {} data in one epoch and actually trained for {} epoch'.format(total_data_per_epoch, cfg.SOLVER.MAX_ITER / total_data_per_epoch)

        logging.getLogger(self.default_log_name).info(format_string)

        return

    def create_dataloader(self, cfg, train_dataset, valid_dataset):
        pin_memory = cfg.TRAINER.DEVICE != 'cpu'
        is_group = cfg.DATALOADER.get('GROUP_SAMPLER', False)
        if cfg.TRAINER.PARADIGM.TYPE == 'DDP' and cfg.TRAINER.PARADIGM.GPU_ID is not None:
            train_data_loader = engine_data_loader.create_distribute_iterable_data_loader(train_dataset,
                                                                                          batch_size=cfg.SOLVER.TRAIN_PER_BATCH,
                                                                                          rank=cfg.TRAINER.PARADIGM.GLOBAL_RANK,
                                                                                          world_size=cfg.TRAINER.PARADIGM.WORLD_SIZE,
                                                                                          num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                                          collate_fn=self.collate_train_fn,
                                                                                          pin_memory=pin_memory,
                                                                                          is_group=is_group)
        else:
            train_data_loader = engine_data_loader.create_iterable_data_loader(train_dataset,
                                                                               batch_size=cfg.SOLVER.TRAIN_PER_BATCH,
                                                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                               collate_fn=self.collate_train_fn,
                                                                               pin_memory=pin_memory,
                                                                               is_group=is_group)

        test_data_loader = engine_data_loader.create_data_loader(valid_dataset,
                                                                 cfg.SOLVER.TEST_PER_BATCH,
                                                                 cfg.DATALOADER.NUM_WORKERS,
                                                                 collate_fn=self.collate_valid_fn,
                                                                 pin_memory=pin_memory)
        return train_data_loader, test_data_loader

    def create_dataset(self, cfg) -> (Dataset, Dataset):
        train_dataset = build_dataset(cfg.DATALOADER.TRAIN_DATA_SET)
        valid_dataset = build_dataset(cfg.DATALOADER.VAL_DATA_SET)

        return train_dataset, valid_dataset

    def create_model(self, cfg) -> BaseModel:
        return build_model(cfg)

    def set_collate_fn(self, cfg):
        pass

    def before_loop(self):
        pass

    def loop(self):
        self.before_loop()

        self.model.enable_train()
        for epoch in range(self.start_iter, self.max_iter):
            if self.enable_epoch_method:
                loss_dict = dict()
                for iteration, data in enumerate(self.train_data_loader):
                    loss_dict = self.model(data, epoch=epoch)
                self.iterate_after(epoch, loss_dict)
                self.checkpoint.save(self.model, epoch)
            else:
                data = next(self.train_data_loader)
                loss_dict = self.model(data, epoch=epoch)
                self.iterate_after(epoch, loss_dict)
                self.checkpoint.save(self.model, epoch)

        self.checkpoint.save(self.model, self.max_iter)

        self.after_loop()
        return

    def after_loop(self):
        pass

    def iterate_after(self, epoch, loss_dict):
        self.checkpoint.save(self.model, epoch)
        if int(epoch + 0.5) % self.checkpoint.check_period == 0:
            logging.getLogger(self.default_log_name).info('trainer run step {} {}'.format(epoch, loss_dict))
        return

    def resume_or_load(self, resume=False):
        """
        :param resume:
        :return:
        """
        model_state_dict, addition_state_dict, start_iter = self.checkpoint.resume_or_load(self.model_path, resume)
        self.start_iter = start_iter
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        if addition_state_dict is not None:
            self.model.load_addition_state_dict(addition_state_dict)
        logging.getLogger(self.default_log_name).info('load model from {}: resume:{} start iter:{}'.format(self.model_path, resume, self.start_iter))
        return
