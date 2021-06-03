import abc


class BaseModel(abc.ABC):
    def __init__(self):
        return

    def save_model(self, save_name, epoch):
        pass

    def load_model(self, load_name):
        pass

    def step(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def enable_model_parallel(self):
        pass

    def enable_model_distributed(self, gpu_id):
        pass

    def enable_train(self):
        pass

    def disable_train(self):
        pass