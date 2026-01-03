import json
import pandas as pd
import networkx as nx

from collections import defaultdict
from .useful_functions import node_ids_to_edge_tuple, get_capacities_from_graph


def read_graph_from_json(topology_filepath):
    """
    Read the graph from a json file.
    Args:
        topology_filepath (str): path to the json file containing the topology
        
    Returns:
        networkx.Graph: graph object
        list: capacities of the links
    """
    # Load the topology data from the JSON file
    with open(topology_filepath) as f:
            data = json.load(f)

    # Convert the JSON data to a NetworkX graph object
    graph = nx.readwrite.json_graph.node_link_graph(data)
    # Extract capacities from the graph
    capacities = get_capacities_from_graph(graph)

    return graph, capacities


def read_tms_from_csv(tms_filepath):
    """
    Read the traffic matrices from a csv file.
    Args:
        tms_filepath (str): path to the csv file containing the traffic matrices

    Returns:
        np.ndarray: traffic matrices
    """
    # Read CSV file into a DataFrame and convert to numpy array
    df = pd.read_csv(tms_filepath, header=None)
    tms = df.values
    # tms = torch.tensor(df.values).float()
    return tms


def read_paths_from_file(filepath, num_nodes, convert=False):
    """Get the paths from the file."""
    # Initialize paths dictionary with default empty lists
    paths = defaultdict(list)
    pid = 0
    
    # Open and read the file containing path information
    with open(filepath, 'r') as f:
        lines = sorted(f.readlines())
        # Create a dictionary mapping source-destination pairs to path lines
        lines_dict = {line.split(":")[0] : line for line in lines if line.strip() != ""}
        
        # Iterate through all possible source-destination pairs
        for src in range(num_nodes):
            for dst in range(num_nodes):
                # Skip if source and destination are the same
                if src == dst:
                    continue
                
                # Check if path exists in the dictionary
                if "(%d, %d)" % (src, dst) in lines_dict:
                    line = lines_dict["(%d, %d)" % (src, dst)].strip()
                else:
                    # Find path line for the current source-destination pair
                    line = [l for l in lines if l.startswith("(%d, %d):" % (src, dst))]
                    if line == []:
                        continue
                    line = line[0]
                    line = line.strip()
                
                # Skip if no line found
                if not line: continue
                
                # Parse the source and destination nodes from the line
                i, j = list(map(int, line.split(":")[0].replace("(", "").replace(")", "").split(", ")))
                
                # Split the line to get the different paths between source and destination
                paths_ = line.split(":")[1].split("; ")
                
                # Process each path between the source and destination
                for p_ in paths_:
                    node_list = list(map( int, p_.split("-")))
                    
                    # Convert node paths to edge paths if convert flag is True
                    if convert:
                        eage_list = node_ids_to_edge_tuple(node_list)
                        paths[(i, j)].append(eage_list)
                    else:
                        paths[(i, j)].append(node_list)
                    pid += 1

    paths = dict(paths)
    return paths
