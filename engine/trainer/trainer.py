import abc
import logging
from engine.model.model import BaseModel
import engine.checkpoint.checkpoint_manager as engine_checkpoint_manager
import engine.comm as comm


class BaseTrainer(abc.ABC):
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
        """
        eg:
            self.model = build_model(cfg)
            self.model.enable_train()
        """
        self.default_log_name = cfg.OUTPUT_LOG_NAME
        self.model = self.create_model(cfg)
        self.model.enable_train()
        self.model.enable_distribute(cfg)

        self.checkpoint = engine_checkpoint_manager.CheckPointerManager(max_iter=cfg.SOLVER.MAX_ITER,
                                                                        save_dir=cfg.OUTPUT_DIR,
                                                                        check_period=cfg.SOLVER.CHECKPOINT_PERIOD,
                                                                        max_keep=cfg.SOLVER.MAX_KEEP,
                                                                        file_prefix=cfg.MODEL.ARCH,
                                                                        save_to_disk=comm.is_main_process())

        self.start_iter = 0
        self.model_path = cfg.MODEL.WEIGHTS
        self.device = cfg.MODEL.DEVICE
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.output = cfg.OUTPUT_DIR
        return

    @abc.abstractmethod
    def create_model(self, cfg) -> BaseModel:
        raise NotImplemented('the create_model must be implement')

    @abc.abstractmethod
    def loop(self):
        raise NotImplemented('the loop must be implement')

    def resume_or_load(self, resume=False):
        """
        :param resume:
        :return:

        eg:
            model_state_dict, addition_state_dict, start_iter = self.checkpoint.resume_or_load(self.model_path, resume)
            self.start_iter = start_iter
            if model_state_dict is not None:
                self.model.load_state_dict(model_state_dict)
            if addition_state_dict is not None:
                self.model.load_addition_state_dict(addition_state_dict)
            logging.getLogger(__name__).info('load model from {}: resume:{} start iter:{}'.format(self.model_path, resume, self.start_iter))
        """
        model_state_dict, addition_state_dict, start_iter = self.checkpoint.resume_or_load(self.model_path, resume)
        self.start_iter = start_iter
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        if addition_state_dict is not None:
            self.model.load_addition_state_dict(addition_state_dict)
        logging.getLogger(self.default_log_name).info('load model from {}: resume:{} start iter:{}'.format(self.model_path, resume, self.start_iter))
        return
