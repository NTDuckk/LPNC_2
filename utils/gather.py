import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all GPUs, preserving gradients for the local GPU only."""

    @staticmethod
    def forward(ctx, input):
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_features(features):
    """Gather features from all GPUs if distributed, else return as-is."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return features
    gathered = GatherLayer.apply(features.contiguous())
    return torch.cat(gathered, dim=0)


def gather_labels(labels):
    """Gather labels from all GPUs (no gradient needed)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return labels
    gathered = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, labels.contiguous())
    return torch.cat(gathered, dim=0)
