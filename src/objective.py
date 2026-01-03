import torch

from .wan_changes import inject_link_failures_by_zero_capacity
from .utils.useful_functions import get_capacities_from_graph, mask_invalid_paths


def compute_mlus(
    x, 
    tm, 
    topology, 
    paths_to_edges, 
    commodities_to_paths, 
    add_failures=False,
    num_failures=1,
    scale=1.
):
    """
    Compute Maximum Link Utilization (MLU) for each sample in the batch.
    
    Args:
        x: Path split ratios of shape [batch_size, num_paths * num_commodities]
        tm: Traffic matrices of shape [batch_size, num_commodities]
        topology: Network topology with capacity information
        paths_to_edges: Matrix mapping paths to edges [num_paths, num_edges]
        commodities_to_paths: Matrix mapping commodities to paths [num_commodities, num_paths]
        add_failures (bool): Whether to simulate link failures
        num_failures (int): Number of link failures to simulate
        scale (float): Scaling factor for capacities
        
    Returns:
        List of MLU values for each sample in the batch
    """
    mlus = []
    B, N = tm.shape  # B: batch size, N: number of commodities

    # Initialize capacity and commodities-to-paths matrix for the case without failures
    if not add_failures:
        # Get capacities from the topology and normalize
        capacity = torch.tensor(get_capacities_from_graph(topology)) / scale
        capacity = capacity.float().unsqueeze(-1).to(tm.device)
        c_2_p = commodities_to_paths  # Use original commodities-to-paths matrix

    # Process each sample in the batch
    for i in range(B):
        if add_failures:
            # Simulate link failures by modifying the topology
            new_topology, _= inject_link_failures_by_zero_capacity(topology, num_failures, seed=i)
            # Get updated capacities after failures
            capacity = torch.tensor(get_capacities_from_graph(new_topology)) / scale
            # Mask invalid paths that use failed edges
            c_2_p = mask_invalid_paths(commodities_to_paths, paths_to_edges, capacity.to(tm.device))
            capacity = capacity.float().unsqueeze(-1).to(tm.device)
            # Set a small value for failed links to avoid division by zero
            capacity[capacity==0] = 1e-2 / scale

        # Extract split ratios and traffic matrix for the current sample
        split_ratios = x[[i]]  # Shape: [1, num_paths * num_commodities]
        true_tm = tm[[i]]      # Shape: [1, num_commodities]
        split_ratios = torch.transpose(split_ratios, 0, 1)  # Shape: [num_paths * num_commodities, 1]

        # Handle failures by redistributing traffic proportionally among remaining tunnels
        # Calculate total weight for each commodity across its paths
        commodity_total_weight = c_2_p.matmul(split_ratios)
        # Calculate redistribution factors for each path
        paths_over_total = c_2_p.transpose(0, 1).matmul(1.0 / commodity_total_weight)
        # Adjust split ratios based on redistribution
        split_ratios = split_ratios.mul(paths_over_total)
        
        # Calculate demand on each path
        tmp_demand_on_paths = c_2_p.transpose(0, 1).matmul(true_tm.transpose(0, 1))  # Shape: (num_paths, 1)
        demand_on_paths = tmp_demand_on_paths.mul(split_ratios)  # Shape: (num_paths, 1)
        # Calculate flow on each edge by summing path demands
        flow_on_edges = paths_to_edges.transpose(0, 1).matmul(demand_on_paths)  # Shape: (num_edges, 1)
        # Calculate congestion on each edge
        congestion = flow_on_edges.divide(capacity)  # Shape: (num_edges, 1)
        # Find the maximum congestion across all edges (MLU)
        max_cong = torch.max(congestion.flatten(), dim = 0).values

        mlus.append(max_cong.item())  # Add the MLU value to the list
    
    return mlus
