import torch.utils.data
import multiprocessing
from engine.data.samplers import TrainingSampler, GroupSampler
from engine.data.common import ToIterableDataset


def create_base_data_loader(dataset, batch_size, num_workers, sampler=None, pin_memory=True, collate_fn=None):
    max_workers = multiprocessing.cpu_count()
    num_workers = num_workers if num_workers < max_workers else max_workers

    return torch.utils.data.DataLoader(
                                        dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        sampler=sampler,
                                        num_workers=num_workers,
                                        pin_memory=pin_memory,
                                        collate_fn=collate_fn,
                                        drop_last=True)


def create_iterable_data_loader(dataset, batch_size, num_workers, collate_fn=None, pin_memory=True, is_group=False):
    if is_group:
        sampler = GroupSampler(dataset, batch_size)
    else:
        sampler = TrainingSampler(len(dataset))
    dataset = ToIterableDataset(dataset, sampler)

    return create_base_data_loader(dataset,
                                   batch_size,
                                   num_workers,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn)


def create_distribute_iterable_data_loader(dataset, batch_size, rank, world_size, num_workers, collate_fn=None, pin_memory=True, is_group=False):
    if is_group:
        sampler = GroupSampler(dataset, batch_size, rank=rank, world_size=world_size)
    else:
        sampler = TrainingSampler(len(dataset), rank=rank, world_size=world_size)
    dataset = ToIterableDataset(dataset, sampler)

    return create_base_data_loader(dataset,
                                   batch_size,
                                   num_workers,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn)


def create_data_loader(dataset, batch_size, num_workers, collate_fn=None, pin_memory=True):

    sampler = torch.utils.data.RandomSampler(dataset)

    return create_base_data_loader(dataset,
                                   batch_size,
                                   num_workers,
                                   sampler=sampler,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn)


def create_distribute_data_loader(dataset, batch_size, rank, world_size, num_workers, collate_fn=None, pin_memory=True):

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    return create_base_data_loader(dataset,
                                   batch_size,
                                   num_workers,
                                   sampler=sampler,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn)

