import math
import torch
import random


def add_temporal_fluctuation_bursts(traffic_demands, scale_factor=10, clip_min=0, clip_max=None, seed=2025):
    """
    Add bursty temporal fluctuation noise to traffic demands using PyTorch.

    Args:
        traffic_demands (torch.Tensor): Tensor of shape [num_samples, time_steps, num_pairs].
        scale_factor (float): Multiplier for the temporal variance.
        clip_min (float or None): Minimum value to clip output.
        clip_max (float or None): Maximum value to clip output.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Noisy traffic of same shape.
    """
    torch.manual_seed(seed)

    # Temporal differences
    diff = traffic_demands[:, 1:, :] - traffic_demands[:, :-1, :]  # shape [N, T-1, P]
    
    # Variance over time axis
    variance = diff.var(dim=1, keepdim=True)  # shape [N, 1, P]
    
    # std = sqrt(scale * var)
    stddev = torch.sqrt(variance * scale_factor)

    # Generate Gaussian noise
    noise = torch.randn_like(traffic_demands) * stddev  # broadcast along time

    # Add noise
    noisy_traffic = traffic_demands + noise

    # Optional clipping
    if clip_min is not None or clip_max is not None:
        noisy_traffic = torch.clamp(noisy_traffic, min=clip_min, max=clip_max)

    return noisy_traffic


def add_spatial_redistribution_bursts(traffic_demands, target_heavy_ratio=0.6, top_k_ratio=0.1):
    """
    Redistribute traffic spatially using PyTorch.
    
    Args:
        traffic_demands (torch.Tensor): Tensor of shape [num_samples, time_steps, num_pairs].
        target_heavy_ratio (float): Target volume ratio for top-k pairs.
        top_k_ratio (float): Fraction of node pairs to be treated as "top-k".
    
    Returns:
        torch.Tensor: Redistributed traffic with same shape.
    """
    num_samples, time_steps, num_pairs = traffic_demands.shape
    device = traffic_demands.device
    redistributed = torch.zeros_like(traffic_demands)

    for i in range(num_samples):
        sample = traffic_demands[i]  # shape: [T, P]
        
        # Total per pair
        total_per_pair = sample.sum(dim=0)  # shape: [P]
        
        # Top-k selection
        top_k = int(num_pairs * top_k_ratio)
        sorted_indices = torch.argsort(total_per_pair, descending=True)
        top_indices = sorted_indices[:top_k]
        rest_indices = sorted_indices[top_k:]

        # Target volumes
        total_volume = total_per_pair.sum()
        target_top_volume = total_volume * target_heavy_ratio
        target_rest_volume = total_volume - target_top_volume

        # Top demands: normalize temporal pattern
        top_original = sample[:, top_indices]  # shape: [T, top_k]
        top_pattern = top_original / (top_original.sum() + 1e-8)
        redistributed_top = top_pattern * target_top_volume

        # Rest demands
        rest_original = sample[:, rest_indices]
        rest_pattern = rest_original / (rest_original.sum() + 1e-8)
        redistributed_rest = rest_pattern * target_rest_volume

        # Assign back
        redistributed[i, :, top_indices] = redistributed_top
        redistributed[i, :, rest_indices] = redistributed_rest

    return redistributed


def shuffle_along_tunnels(x: torch.Tensor, K: int, return_indices: bool = False):
    """
    Args:
        x: (K*M, D) Input tensor, considered as the concatenation of M (K, D) tensors.
        K: Number of the shortest paths.
        return_indices: Whether to return the shuffle indices (used for recovery).
    
    Returns:
        shuffled_x: (K*M, D) Shuffled tensor.
        indices: (M, K) Shuffle indices (only returned if return_indices=True).
    """
    M = x.size(0) // K 
    D = x.size(1)
    
    x_split = x.view(M, K, D)
    
    # randomly shuffle along the K dimension
    indices = torch.stack([torch.randperm(K) for _ in range(M)], dim=0).to(x.device)
    # shuffled_x1 = torch.stack([x_split[i, indices[i]] for i in range(M)])
    shuffled_x = x_split[torch.arange(M).unsqueeze(-1), indices]
    # print(shuffled_x1 == shuffled_x)
    
    shuffled_x = shuffled_x.view(-1, D)
    
    if return_indices:
        return shuffled_x, indices
    else:
        return shuffled_x
    

def unshuffle_along_tunnels(shuffled_x: torch.Tensor, indices: torch.Tensor, K: int):
    """
    Args:
        shuffled_x: (K*M, D) Shuffled tensor.
        indices: (M, K) Shuffle indices (used for recovery).
        K: Number of the shortest paths.
    
    Returns:
        original_x: (K*M, D) Tensor restored to its original order.
    """
    M = shuffled_x.size(0) // K
    D = shuffled_x.size(1)
    
    x_split = shuffled_x.view(M, K, D)
    
    # restore the original order using the indices
    inverse_indices = torch.argsort(indices, dim=1).to(x_split.device)  # Get the inverse indices for unshuffling
    # original_x1 = torch.stack([x_split[i, inverse_indices[i]] for i in range(M)])
    original_x = x_split[torch.arange(M).unsqueeze(-1), inverse_indices]
    # print(original_x1 == original_x)
    
    original_x = original_x.view(-1, D)
    
    return original_x


def inject_link_failures_by_zero_capacity(graph, num_failures=1, seed=42):
    """
    Randomly set the capacity of selected links to zero to simulate link failures.

    Args:
        graph (networkx.Graph): Original topology.
        num_failures (int): Number of links to fail.
        seed (int): Random seed.

    Returns:
        networkx.Graph: Modified graph (in-place).
        list of tuple: Failed links.
    """

    random.seed(seed)

    edges = list(graph.edges())
    failed_links = random.sample(edges, num_failures)

    for u, v in failed_links:
        graph[u][v]['capacity'] = 0

    return graph, failed_links 
