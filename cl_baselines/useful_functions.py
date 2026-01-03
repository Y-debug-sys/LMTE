import torch
import numpy as np


def Get_path_capacities(graph, paths):
    """
    Calculate and return the capacity of each path based on the minimum edge capacity in the path.
    
    Args:
        graph: NetworkX graph object with edge capacities
        paths: Dictionary containing paths between node pairs, paths[(src, dst)] = [path_list]
    
    Returns:
        dict: A dictionary mapping each (src, dst, path_index) tuple to its path capacity,
              which is the minimum capacity of all edges in that specific path
    """
    path_capacity = {}
    # Iterate through all possible source and destination pairs
    for src in range(graph.number_of_nodes()):
        for dst in range(graph.number_of_nodes()):
            if dst != src:
                # For each path between src and dst, calculate its capacity
                # Path capacity is the minimum capacity among all edges in the path
                for index, path in enumerate(paths[(src, dst)]):
                    path_capacity[(src, dst, index)] = min([graph.edges[(path[i], path[i + 1])]['capacity'] \
                                         for i in range(len(path) - 1)])
    return path_capacity


def Get_edge_to_path(topology, candidate_path):
    """
    Create a mapping from each edge to the paths that use that edge.
    
    Args:
        topology: NetworkX graph object representing the network topology
        candidate_path: Dictionary containing candidate paths between node pairs
    
    Returns:
        dict: A dictionary mapping each edge (u, v) to a list of paths that use this edge,
              where each path is represented as (src, dst, path_index)
    """
    # Initialize edge_to_path dictionary with empty lists for each edge
    edge_to_path = {}
    for edge in topology.edges:
        edge_to_path[(int(edge[0]),int(edge[1]))] = []
    
    # For each source-destination pair and each of their paths, add the path to the list
    # of paths using each edge in the path
    for src in topology.nodes:
        for dst in topology.nodes:
            if src != dst:
                for index, path in enumerate(candidate_path[(src, dst)]):
                    # Add the path to the list of paths that use each edge in the path
                    for i in range(len(path) - 1):
                        edge_to_path[(int(path[i]), int(path[i + 1]))].append((int(src), int(dst), index))
    return edge_to_path


def get_weight_dict_tensor(path_weight_routing, num_nodes, num_paths):
    """
    Convert a possibly incomplete path_weight_routing dict into a complete 1D tensor,
    ensuring all 'w_i_j_k' (for i != j, k in 0..num_paths-1) are present.

    Args:
        path_weight_routing (dict): partial mapping from 'w_i_j_k' to weight value
        num_nodes (int): total number of nodes
        num_paths (int): number of candidate paths per (i, j) pair

    Returns:
        tensor: 1D tensor of weights in fixed order
        keys: list of corresponding keys
    """

    keys = list(path_weight_routing.keys())  # preserve order
    values = [path_weight_routing[k] for k in keys]
    tensor = torch.tensor(values).float()
    return tensor


def expand_tms(tensor, num_nodes):
    """
    Expand a NumPy tensor of shape [..., num_nodes * (num_nodes - 1)] to
    [..., num_nodes * num_nodes], inserting 0 at self-loop positions (i == j).

    Args:
        tensor (np.ndarray): input tensor with no self-loops in the last dim
        num_nodes (int): number of nodes

    Returns:
        np.ndarray: expanded tensor with self-loop positions set to 0
    """
    # Get the original shape excluding the last two dimensions
    orig_shape = tensor.shape[:-1]
    # Create output tensor with shape [..., num_nodes, num_nodes]
    out = np.zeros((*orig_shape, num_nodes, num_nodes), dtype=tensor.dtype)

    # Build index pairs for non-self-loop entries (i ≠ j)
    idx = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    idx = np.array(idx)

    # Assign values to the appropriate positions in the output tensor
    for n in range(len(idx)):
        i, j = idx[n]
        out[..., i, j] = tensor[..., n]

    # Return tensor with the last two dimensions flattened
    return out.reshape(*orig_shape, num_nodes * num_nodes)


def remove_self_loops(tensor, num_nodes, num_paths):
    """
    Given a tensor of shape (num_nodes * num_nodes * num_paths,),
    remove entries where i == j, and return shape (num_nodes * (num_nodes - 1) * num_paths,)

    Args:
        tensor (torch.Tensor): shape (num_nodes * num_nodes * num_paths,)
        num_nodes (int)
        num_paths (int)

    Returns:
        torch.Tensor: filtered tensor
    """
    # Get the device of the input tensor to ensure operations happen on the same device
    device = tensor.device

    # Build i, j, k index grid
    # i represents source nodes, repeated for all destination and path combinations
    i = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes * num_paths)
    # j represents destination nodes, repeated in a cycle for all sources and paths
    j = torch.arange(num_nodes, device=device).repeat(num_nodes * num_paths)
    # k represents path indices, repeated in a cycle for all source-destination pairs
    k = torch.arange(num_paths, device=device).repeat(num_nodes * num_nodes)

    # Build mask to filter out cases where source equals destination (i == j)
    mask = (i != j)

    # Apply mask to flatten input, keeping only entries where i ≠ j
    return tensor[mask]