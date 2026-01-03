import os
import torch
import shutil
import numpy as np
import networkx as nx

from itertools import islice
from scipy.sparse import csr_matrix, lil_matrix


def get_capacities_from_graph(graph):
    """Extract capacities from the graph edges.
    
    Args:
        graph (networkx.Graph): Input graph with capacity attribute on edges
        
    Returns:
        list: List of capacities for each edge in the graph
    """
    capacities = [float(data['capacity']) for u, v, data in graph.edges(data=True)]
    return capacities


def node_ids_to_edge_tuple(node_ids):
    """Convert the node list path to the edge list path."""
    return [(node1, node2) for node1, node2 in zip(node_ids, node_ids[1:])]


def del_files(dir_path):
    """Delete all files in the directory."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def get_paths_to_edges(topology, paths):
    """Get the paths_to_edges matirx, [num_paths, num_edges]
       paths_to_edges[i, j] = 1 if edge j is in path i
    """
    # Get number of nodes and edges in the topology
    num_nodes = topology.number_of_nodes()
    num_edges = topology.number_of_edges()
    paths_arr = []

    # Create adjacency matrix for the topology
    adj = np.zeros((num_nodes, num_nodes))
    for s in range(num_nodes):
        for d in range(num_nodes):
            if s == d:
                continue
            if d in topology[s]:
                adj[s,d] = 1

    # Create mapping from edges to unique IDs
    eid = 0
    edges_map = dict()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i,j] == 1:
                edges_map[(i, j)] = eid
                eid += 1
    
    # For each path, create a binary vector indicating which edges are used
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            for p in paths[(i, j)]:
                # Map each edge in the path to its ID
                p_ = [edges_map[e] for e in p]
                # Create a binary vector for this path
                p__ = np.zeros((int(num_edges),))
                for k in p_:
                    p__[k] = 1
                paths_arr.append(p__)

    # Return sparse matrix representation of paths to edges mapping
    return csr_matrix(np.stack(paths_arr))


def get_commodities_to_paths(topology, num_paths, paths):
    """Get the commodities_to_paths matrix, [num_commodities, num_paths]
       commodities_to_paths[i, j] = 1 if path j is a candidate path for commodity i
    """
    # Initialize sparse matrix for commodities to paths mapping
    num_nodes = topology.number_of_nodes()
    commodities_to_paths = lil_matrix((num_nodes * (num_nodes - 1), num_paths))
    commid = 0
    pathid = 0
    
    # Populate the matrix for each source-destination pair
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue
            # For each path between src and dst, mark it in the matrix
            for _ in paths[(src, dst)]:
                commodities_to_paths[commid, pathid] = 1
                pathid += 1
            commid += 1
    return csr_matrix(commodities_to_paths)



def mask_invalid_paths(commodities_to_paths, paths_to_edges, capacities):
    """
    Mask out invalid paths in commodities_to_paths where the path uses an edge with capacity == 0.

    Args:
        commodities_to_paths (torch.Tensor): [num_commodities, num_paths], 0/1
        paths_to_edges (torch.Tensor): [num_paths, num_edges], 0/1
        capacities (torch.Tensor): [num_edges,], float or int, where 0 means edge failure

    Returns:
        torch.Tensor: modified commodities_to_paths with invalid paths zeroed out
    """
    # Step 1: find failed edges
    failed_edges = (capacities == 0).float()  # [num_edges]

    # Step 2: find invalid paths (any path using any failed edge)
    invalid_paths_mask = (paths_to_edges @ failed_edges) > 0  # [num_paths], bool

    # Step 3: mask out columns in commodities_to_paths
    # Turn it into a [1, num_paths] mask for broadcasting
    valid_paths_mask = (~invalid_paths_mask).float().unsqueeze(0)  # [1, num_paths]

    # Step 4: zero out invalid paths
    masked_commodities_to_paths = commodities_to_paths * valid_paths_mask  # broadcasted over rows

    return masked_commodities_to_paths


def compute_ksp_paths(k, pairs, graph, save2txt=False, filepath=None, transform=False):
    """
    Computes or loads the k-shortest paths for each source-destination pair.

    Args:
        k (int): The number of shortest paths to compute for each pair.
        pairs (Iterable (np.array, list, OrderedSet... etc.)): A list of source-destination node pairs (src, dst).
        graph (networkx.Graph): The graph to compute the paths on.
        save2pkl (bool): Whether to save the computed paths to a pickle file.
        filepath (str): The path to save the pickle file.
        transform (bool): Whether to transform the paths from edge tuples to node lists.

    Returns:
        dict: A dictionary where the keys are (src, dst) tuples and the values are lists of 
            k paths represented as edge tuples.

    Process:
        1. Compute the k-shortest paths for each source-destination pair in the given pairs list.
        - Use `networkx.shortest_simple_paths` to compute the paths.
        - If a pair has fewer than k paths, replicate the first path until the number of paths equals k.
        - Store the paths as edge tuples by converting node IDs using `node_ids_to_edge_tuple`.
        3. Save the computed k-shortest paths to the pickle file for future use.
        4. Return the dictionary containing the k-shortest paths.
    """
    # Initialize paths dictionary and start computation
    paths = dict()
    print(f"[Computing {k} Shortest Paths]")
    for src, dst in pairs:
        # Compute k shortest paths between the source and destination
        all_paths = list(islice(nx.shortest_simple_paths(graph, src, dst, weight=None), k))
        
        # Ensure each pair has exactly k paths by replicating if needed
        while len(all_paths) != k:
            all_paths.append(all_paths[0])
        
        # Convert node paths to edge tuples
        paths[(src, dst)] = [node_ids_to_edge_tuple(all_paths[i]) for i in range(k)]

    # Save paths to text file if requested
    if save2txt:
        # Save the paths to a text file, such that each line contains the top-k shortest paths for a pair of nodes.
        with open(os.path.join(filepath, "paths.txt"), 'w') as file:
            for (src, dst), path_list in paths.items():
                # Convert edge tuples to string representation
                path_strs = ['-'.join(map(str, [edge[0] for edge in path] + [path[-1][1]])) for path in path_list]
                file.write(f"({src}, {dst}): {'; '.join(path_strs)}\n")
            file.close()

    # Transform paths from edge tuples to node lists if requested
    if transform:
        # Transform the paths from edge tuples to node lists
        transformed_paths = {}
        for (src, dst), path_list in paths.items():
            transformed_paths[(src, dst)] = [[edge[0] for edge in path] + [path[-1][1]] for path in path_list]
            
        return transformed_paths
    
    return paths


def random_tensor_with_ratio(shape, ratio=0.5, val1=1, val2=2):
    """
    Generate a tensor of the given shape with random positions filled with val1 and val2,
    according to the specified ratio.

    Args:
        shape (tuple): Shape of the output tensor, e.g., (10, 10)
        ratio (float): Fraction of elements to be val1 (between 0 and 1)
        val1 (int or float): Value to assign to 'ratio' fraction of elements
        val2 (int or float): Value to assign to the rest of the elements
        dtype (torch.dtype): Data type of the tensor (default: torch.int)
        device (str or torch.device): Device to place the tensor on (default: 'cpu')

    Returns:
        torch.Tensor: A tensor with randomly assigned val1 and val2 values
    """
    # Calculate number of elements to assign as val1
    num_elements = torch.tensor(shape).prod().item()
    num_val1 = int(num_elements * ratio)

    # Create a flat tensor filled with val2
    tensor = torch.full((num_elements,), val2)

    # Randomly select indices to assign val1
    indices = torch.randperm(num_elements)[:num_val1]
    tensor[indices] = val1

    # Reshape to the desired shape and return
    return tensor.view(shape).float()