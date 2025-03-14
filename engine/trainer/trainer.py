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
        self.start_iter = 0
        self.model_path = cfg.TRAINER.WEIGHTS
        self.device = cfg.TRAINER.DEVICE
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.output = cfg.OUTPUT_DIR
        self.enable_epoch_method = cfg.TRAINER.ENABLE_EPOCH_METHOD

        self.gradient_accumulation_batch = int(cfg.SOLVER.GRADIENT_ACCUMULATION_BATCH)

        if self.gradient_accumulation_batch > 1:
            self.train_batch_size = cfg.SOLVER.TRAIN_PER_BATCH // self.gradient_accumulation_batch
            assert self.train_batch_size * self.gradient_accumulation_batch == cfg.SOLVER.TRAIN_PER_BATCH, 'the TRAIN_PER_BATCH must be divisible by GRADIENT_ACCUMULATION_BATCH'
        else:
            self.train_batch_size = cfg.SOLVER.TRAIN_PER_BATCH

        self.model = self.create_model(cfg)
        self.model.enable_train()
        self.model.enable_distribute(cfg)

        if hasattr(self.model.g_model, 'model_name'):
            model_name = self.model.g_model.model_name
        else:
            model_name = self.model.g_model.__class__.__name__
            if isinstance(self.model.g_model,
                          (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
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

        train_dataset, valid_dataset, test_dataset = self.create_dataset(cfg)
        total_data_per_epoch = len(train_dataset) / self.train_batch_size
        if self.enable_epoch_method:
            train_data_loader, val_data_loader, test_data_loader = self.create_dataloader(cfg, train_dataset,
                                                                                          valid_dataset, test_dataset)
            actual_epoch = self.max_iter
        else:
            train_data_loader, val_data_loader, test_data_loader = self.create_iterable_dataloader(cfg, train_dataset,
                                                                                                   valid_dataset,
                                                                                                   test_dataset)
            train_data_loader = iter(train_data_loader)
            actual_epoch = self.max_iter / total_data_per_epoch

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        format_string = 'create train dataset {}\n'.format(train_dataset)
        format_string += 'create valid dataset {}\n'.format(valid_dataset)
        format_string += 'create test dataset {}\n'.format(test_dataset)
        format_string += 'enable gradient accumulation {}, train_batch_size={}\n'.format(
            self.gradient_accumulation_batch, self.train_batch_size)
        format_string += 'there are {} data in one epoch and actually trained for {} epoch'.format(total_data_per_epoch,
                                                                                                   actual_epoch)

        logging.getLogger(self.default_log_name).info(format_string)

        return

    def create_iterable_dataloader(self, cfg, train_dataset, valid_dataset, test_dataset):
        pin_memory = cfg.TRAINER.DEVICE != 'cpu'
        is_group = cfg.DATALOADER.get('GROUP_SAMPLER', False)
        if cfg.TRAINER.PARADIGM.TYPE == 'DDP' and cfg.TRAINER.PARADIGM.GPU_ID is not None:
            train_data_loader = engine_data_loader.create_distribute_iterable_data_loader(train_dataset,
                                                                                          batch_size=self.train_batch_size,
                                                                                          rank=cfg.TRAINER.PARADIGM.GLOBAL_RANK,
                                                                                          world_size=cfg.TRAINER.PARADIGM.WORLD_SIZE,
                                                                                          num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                                          collate_fn=self.collate_train_fn,
                                                                                          pin_memory=pin_memory,
                                                                                          is_group=is_group)
        else:
            train_data_loader = engine_data_loader.create_iterable_data_loader(train_dataset,
                                                                               batch_size=self.train_batch_size,
                                                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                               collate_fn=self.collate_train_fn,
                                                                               pin_memory=pin_memory,
                                                                               is_group=is_group)

        val_data_loader = engine_data_loader.create_data_loader(valid_dataset,
                                                                cfg.SOLVER.VAL_PER_BATCH,
                                                                num_workers=1,
                                                                collate_fn=self.collate_valid_fn,
                                                                pin_memory=pin_memory)

        test_data_loader = engine_data_loader.create_data_loader(test_dataset,
                                                                 cfg.SOLVER.TEST_PER_BATCH,
                                                                 num_workers=1,
                                                                 collate_fn=self.collate_valid_fn,
                                                                 pin_memory=pin_memory)

        return train_data_loader, val_data_loader, test_data_loader

    def create_dataloader(self, cfg, train_dataset, valid_dataset, test_dataset):
        pin_memory = cfg.TRAINER.DEVICE != 'cpu'
        if cfg.TRAINER.PARADIGM.TYPE == 'DDP' and cfg.TRAINER.PARADIGM.GPU_ID is not None:
            train_data_loader = engine_data_loader.create_distribute_data_loader(train_dataset,
                                                                                 batch_size=self.train_batch_size,
                                                                                 rank=cfg.TRAINER.PARADIGM.GLOBAL_RANK,
                                                                                 world_size=cfg.TRAINER.PARADIGM.WORLD_SIZE,
                                                                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                                 collate_fn=self.collate_train_fn,
                                                                                 pin_memory=pin_memory)
        else:
            train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.train_batch_size,
                                                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                            shuffle=True,
                                                            drop_last=False,
                                                            collate_fn=self.collate_train_fn,
                                                            pin_memory=pin_memory)

        val_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                      cfg.SOLVER.VAL_PER_BATCH,
                                                      num_workers=1,
                                                      collate_fn=self.collate_valid_fn,
                                                      pin_memory=pin_memory)

        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       cfg.SOLVER.TEST_PER_BATCH,
                                                       num_workers=1,
                                                       collate_fn=self.collate_valid_fn,
                                                       pin_memory=pin_memory)
        return train_data_loader, val_data_loader, test_data_loader

    def create_dataset(self, cfg) -> (Dataset, Dataset, Dataset):
        train_dataset = build_dataset(cfg.DATALOADER.TRAIN_DATA_SET)
        valid_dataset = build_dataset(cfg.DATALOADER.VAL_DATA_SET)
        if cfg.DATALOADER.get('TEST_DATA_SET', None) is not None:
            test_dataset = build_dataset(cfg.DATALOADER.TEST_DATA_SET)
            logging.getLogger(self.default_log_name).info(
                f'create test dataset by {cfg.DATALOADER.TEST_DATA_SET.dump()}')
        else:
            test_dataset = valid_dataset

        return train_dataset, valid_dataset, test_dataset

    def create_model(self, cfg) -> BaseModel:
        return build_model(cfg)

    def set_collate_fn(self, cfg):
        pass

    def before_loop(self):
        pass

    def loop(self):
        self.before_loop()

        self.model.enable_train()

        step_info = dict()
        for epoch in range(self.start_iter, self.max_iter):
            if self.enable_epoch_method:
                for iteration, data in enumerate(self.train_data_loader):
                    step_info = self.model(data, epoch=epoch, data_epoch=iteration,
                                           accumulation_epoch=self.gradient_accumulation_batch)

                self.checkpoint.save(self.model, epoch)
            else:
                data = next(self.train_data_loader)
                step_info = self.model(data, epoch=epoch, data_epoch=epoch,
                                       accumulation_epoch=self.gradient_accumulation_batch)

                if self.gradient_accumulation_batch < 1:
                    self.checkpoint.save(self.model, epoch)
                else:
                    if 0 == (epoch + 1) % self.gradient_accumulation_batch:
                        self.checkpoint.save(self.model, epoch)

            if epoch % self.checkpoint.check_period == 0:
                logging.getLogger(self.default_log_name).info('trainer run step {} {}'.format(epoch, step_info))

            self.iterate_after(epoch)

        self.checkpoint.save(self.model, self.max_iter)

        self.after_loop()
        return

    def after_loop(self):
        pass

    def iterate_after(self, epoch):
        pass

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
        logging.getLogger(self.default_log_name).info(
            'load model from {}: resume:{} start iter:{}'.format(self.model_path, resume, self.start_iter))
        return
