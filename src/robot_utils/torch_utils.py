import io
import boto3
import torch
import os
from typing import Any, Callable
from collections.abc import Iterator
import torch.nn as nn
import numpy as np

def torch_load(data_path: str, **kwargs):
    if data_path.startswith("s3://"):
        s3 = boto3.client("s3")
        buffer = io.BytesIO()
        bucket_name, key = data_path.replace("s3://", "").split("/", 1)
        s3.download_fileobj(bucket_name, key, buffer)
        buffer.seek(0)
        data = torch.load(buffer, **kwargs)
    else:
        data = torch.load(data_path, **kwargs)
    return data

def torch_save(data: Any, path: str | io.BufferedWriter, **kwargs):
    if isinstance(path, str) and path.startswith("s3://"):
        s3 = boto3.client("s3")
        buffer = io.BytesIO()
        torch.save(data, buffer, **kwargs)
        buffer.seek(0)
        bucket_name, key = path.replace("s3://", "").split("/", 1)
        s3.upload_fileobj(buffer, bucket_name, key)
    else:
        if isinstance(path, str) and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(data, path, **kwargs)

def filter_params(
    named_params: Iterator[tuple[str, nn.Parameter]],
    keywords: list[str] | None,
    requires_grad: bool | None = None,
) -> Iterator[tuple[str, nn.Parameter]]:
    if keywords is not None:
        keywords_used = np.zeros(len(keywords), dtype=bool)
    else:
        keywords_used = None

    for i, (name, param) in enumerate(named_params):
        if keywords is None or any(keyword in name for keyword in keywords):
            if keywords_used is not None and keywords is not None:
                keywords_used = keywords_used | np.array(
                    [keyword in name for keyword in keywords]
                )

            if requires_grad is not None:
                if param.requires_grad == requires_grad:
                    yield name, param
            else:
                yield name, param
    if keywords_used is not None and keywords is not None:
        for i, keyword in enumerate(keywords):
            if not keywords_used[i]:
                raise ValueError(f"Keyword {keyword} not used! Please check the names")


def exclude_params(
    named_params: Iterator[tuple[str, nn.Parameter]],
    keywords: list[str],
    requires_grad: bool | None = None,
) -> Iterator[tuple[str, nn.Parameter]]:
    for name, param in named_params:
        if not any(keyword in name for keyword in keywords):
            if requires_grad is not None:
                if param.requires_grad == requires_grad:
                    yield name, param
            else:
                yield name, param


def params(named_params: Iterator[tuple[str, nn.Parameter]]) -> Iterator[nn.Parameter]:
    for _, param in named_params:
        yield param


def aggregate_batch(
    batch: list[Any], aggregate_fn: Callable[[list[Any]], Any], merge_none: bool = True
) -> Any:
    """
    Custom collate function to concatenate nested tensors/ndarray/float along a specified axis.
    If merge_none is True, the field that has None values will be merged into a single None value. Otherwise will return a list of None values.
    Popular choices of aggregate_fn:
        - partial(torch.cat, dim=existing_dim), if you want to concatenate along an existing dimension
        - partial(torch.stack, dim=new_dim), if you want to stack to a new dimension

    Args:
        batch (List[Any]): A list of samples from the dataset.
        aggregate_fn (Callable[[list[Any]], Any]): The function to aggregate the tensors/ndarray/float.

    Returns:
        Any: The concatenated batch.
    """
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if (
        isinstance(elem, torch.Tensor)
        or isinstance(elem, np.ndarray)
        or isinstance(elem, float)
    ):
        return aggregate_fn(batch)
    elif isinstance(elem, dict):
        return {
            key: aggregate_batch([d[key] for d in batch], aggregate_fn)
            for key in elem.keys()
        }
    elif isinstance(elem, list):
        return [aggregate_batch(samples, aggregate_fn) for samples in zip(*batch)]
    elif elem is None:
        if merge_none:
            return None
    else:
        return batch


def split_batch(
    batch: Any, split_fn: Callable[[torch.Tensor], tuple[torch.Tensor, ...]]
) -> Iterator[Any]:
    """
    Split a batch into multiple batches along a specified dimension.
    Popular choices of split_fn:
        - partial(torch.split, dim=existing_dim), if you want to split along an existing dimension
        - partial(torch.unbind, dim=diminishing_dim), if you want to split and diminish a dimension

    Args:
        batch (Any): Should be a nested dict or a nested list, where all the elements are tensors.
        split_fn (Callable[[torch.Tensor], tuple[torch.Tensor, ...]]): The function to split the batch.

    Returns:
        Iterator[Any]: An iterator over the split batches.
    """
    if isinstance(batch, torch.Tensor):
        yield from split_fn(batch)

    elif isinstance(batch, dict):
        for values in zip(*[split_batch(v, split_fn) for v in batch.values()]):
            yield {k: v for k, v in zip(batch.keys(), values)}

    elif isinstance(batch, list):
        for values in zip(*[split_batch(v, split_fn) for v in batch]):
            yield [v for v in values]

    else:
        raise ValueError(f"Invalid batch type: {type(batch)}")


def to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(item) for item in obj]
    else:
        return obj

process_group_initialized = False

world_size = int(os.environ.get("WORLD_SIZE", 1))

def init_process_group():
    global process_group_initialized
    if not process_group_initialized and world_size > 1:
        print(f"Environment variable WORLD_SIZE={os.environ.get('WORLD_SIZE')}, RANK={os.environ.get('RANK')}. Using distributed training.")
        torch.distributed.init_process_group(backend="nccl")
        process_group_initialized = True

def is_main_process():
    init_process_group()
    if torch.distributed.is_initialized():
        return int(os.environ.get("RANK", 0)) == 0
    else:
        return True

def wait_for_main_process():
    init_process_group()
    if torch.distributed.is_initialized():
        # BUG: This will lead to memory leak on GPU 0. Not sure which part of the code is causing this.
        # Do not use this function before accelerator is initialized.
        torch.distributed.barrier()
    else:
        print("Not using distributed training. No need to wait for main process.")
    
def num_processes():
    init_process_group()
    if torch.distributed.is_initialized():
        return int(os.environ.get("WORLD_SIZE", 1))
    else:
        return 1