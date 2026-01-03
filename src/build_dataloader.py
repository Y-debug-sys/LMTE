import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.utils.read_data import read_graph_from_json, read_tms_from_csv


def build_dataloader(
    topology_filename, 
    # opts_filename, 
    tms_filename, 
    batch_size, 
    scale=10**9, 
    eval_batch_size=None, 
    window_size=12, 
    split_ratio=(0.7, 0.1, 0.2)
):
    """ Build dataloaders for training, validation and testing.

    Args:
        topology_filename (str): path to the json file containing the topology
        # opts_filename (str): path to the txt file containing the optimal values
        tms_filename (str): path to the csv file containing the traffic matrices
        batch_size (int): batch size
        window_size (int): size of the sliding window
        split_ratio (tuple): ratio to split the dataset into training, validation and testing sets
    
    Returns:
        tuple: training, validation and testing dataloaders
    """

    # Set evaluation batch size to training batch size if not provided
    eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
    _, valid_ratio, test_ratio = split_ratio
    
    # Create train dataset and dataloader
    train_dataset = TEDataset(topology_filename, tms_filename, window_size, scale=scale,
                              mode='train', test_ratio=test_ratio, valid_ratio=valid_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create validation dataset and dataloader
    valid_dataset = TEDataset(topology_filename, tms_filename, window_size, scale=scale,
                              mode='valid', test_ratio=test_ratio, valid_ratio=valid_ratio)
    valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=True)
    
    # Create test dataset and dataloader
    test_dataset = TEDataset(topology_filename, tms_filename, window_size, scale=scale,
                             mode='test', test_ratio=test_ratio, valid_ratio=valid_ratio)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


class TEDataset(Dataset):

    """ Traffic Engineering Dataset. 

    Args:
        topology_filename (str): path to the json file containing the topology
        # opts_filename (str): path to the txt file containing the optimal values
        tms_filename (str): path to the csv file containing the traffic matrices
        window_size (int): size of the sliding window
        mode (str): mode of the dataset (train, valid, test)
        test_ratio (float): ratio of the test set
        valid_ratio (float): ratio of the validation set
    """

    def __init__(
        self,
        topology_filename, 
        # opts_filename, 
        tms_filename,
        window_size=12,
        mode='train',
        test_ratio=0.2,
        valid_ratio=0.1,
        scale=10**9
    ):
        super(TEDataset, self).__init__()

        # Store normalization scale factor
        self.scale = scale
        # Read topology and capacities from JSON file and normalize capacities
        self.topo, self.capacities = read_graph_from_json(topology_filename)
        self.capacities = self.normalize(torch.tensor(self.capacities))
        
        # Initialize variables for edge index and node features
        self.edge_index = None 
        self.node_features = None 
        self.edge_ids_per_path = None 
        self.mandatory_edges = None
        # self.node_features = self.get_node_features() 

        # Create a mapping from edges to their IDs for quick lookup
        self.edges_map = {(i, j): eid for eid, (i, j) in enumerate(self.topo.edges())}

        # Create a mask to exclude diagonal (self-loop) traffic matrix entries
        num_nodes = len(self.topo.nodes())
        tm_mask = np.ones((num_nodes, num_nodes), dtype=bool)
        np.fill_diagonal(tm_mask, 0)
        self.tm_mask = tm_mask.flatten()

        # Load and process traffic matrices
        # Normalizes traffic matrices and flattens OD pairs
        tms = self.normalize(read_tms_from_csv(tms_filename)[:, self.tm_mask])
        # Prepare target traffic matrices (shifted by one time step)
        tm_preds = torch.from_numpy(tms[window_size:, :])
        # Create sequences of traffic matrices using sliding window
        tm_seqences = np.array(self.sample_silding_windows(tms, window_size))
        tm_seqences = torch.from_numpy(tm_seqences)

        # Split the dataset based on mode (train, valid, test)
        if mode == 'train':
            # Training data: from start to (1 - test_ratio - valid_ratio)
            self.train_tms = tms[:int((1 - test_ratio - valid_ratio) * len(tm_seqences))]
            self.tm_seqences = tm_seqences[:int((1 - test_ratio - valid_ratio) * len(tm_seqences)), :]
            self.tm_preds = tm_preds[:int((1 - test_ratio - valid_ratio) * len(tm_seqences)), :]
            # self.optimal_values = optimal_values[:int((1 - test_ratio - valid_ratio) * len(tm_seqences))]
        elif mode == 'test':
            # Testing data: from end backwards by test_ratio
            self.tm_seqences = tm_seqences[- int(test_ratio * len(tm_seqences)):, :]
            self.tm_preds = tm_preds[- int(test_ratio * len(tm_seqences)):, :]
            # self.optimal_values = optimal_values[- int(test_ratio * len(tm_seqences)):]
        elif mode == 'valid':
            # Validation data: between test and train portions
            self.tm_seqences = tm_seqences[- int((valid_ratio + test_ratio) * len(tm_seqences)): - int(test_ratio * len(tm_seqences)), :]
            self.tm_preds = tm_preds[- int((valid_ratio + test_ratio) * len(tm_seqences)): - int(test_ratio * len(tm_seqences)), :]
            # self.optimal_values = optimal_values[- int((valid_ratio + test_ratio) * len(tm_seqences)): - int(test_ratio * len(tm_seqences))]
        else:
            raise ValueError(f"Invalid mode: {mode}. Please choose between 'train', 'valid' and 'test'.")

    def get_padded_edge_ids_per_path(self, pij) -> torch.Tensor:
        """
        Retrieves or computes the padded edge IDs for each path in pij.

        Args:
            pij (dict): Dictionary where the keys are (src, dst) pairs and values are lists of paths 
                        represented as edge tuples.

        Returns:
            torch.Tensor: A tensor of padded edge IDs for each path. The tensor is padded with -1 
                        to ensure all paths have the same length, and has shape 
                        (num_paths, max_path_length).

        Process:
            1. Compute the edge IDs for each path:
            - For each (src, dst) pair in pij, retrieve the list of paths.
            - For each path, convert its edges to indices using the edges_map.
            - Append the edge indices to a list.
            2. Pad the edge index lists so that all paths have the same length, using -1 as the 
            padding value.
            3. Convert the padded edge indices to a tensor of int64 type and save it to the file.
            4. Return the padded edge IDs tensor.
        """

        # Return cached result if already computed
        if self.edge_ids_per_path is not None:
            return self.edge_ids_per_path

        # Process each path in the input dictionary to convert edges to IDs
        paths_edges_list = []
        for key in pij.keys():
            for path in pij[key]:
                edges_list = []
                for edge in path:
                    index = self.edges_map[edge]
                    edges_list.append(index)
                paths_edges_list.append(torch.tensor(edges_list, dtype=torch.int32))
                
        # Pad sequences to the same length and convert to tensor
        padded_edge_ids_per_path = torch.nn.utils.rnn.pad_sequence(paths_edges_list, batch_first=True,
                                                                   padding_value=-1.0)
        padded_edge_ids_per_path = padded_edge_ids_per_path.to(dtype=torch.int64)
        
        self.edge_ids_per_path = padded_edge_ids_per_path
        return padded_edge_ids_per_path
    
    def get_node_features(self, topo=None, capacities=None):
        """
        Computes and returns the node features for the graph.

        The node features are composed of:
        1. Node degrees: The in-degree of each node in the graph.
        2. Capacity sums: The sum of capacities of edges connected to each node.
        
        The function performs the following steps:
        - Computes the in-degrees of all nodes in the graph and reshapes them.
        - Retrieves the edge index of the graph.
        - Adjusts the capacities to avoid zero values based on the mode (train or test).
        - Computes the sum of capacities for edges connected to each node.
        - Concatenates the node degrees and capacity sums to form the node features.
        
        Returns:
            torch.Tensor: A tensor containing the node features with shape (num_nodes, 2).
        """

        # if self.node_features is not None:
        #     return self.node_features

        topo = self.topo if topo is None else topo

        capacities = self.capacities if capacities is None else capacities
        
        # Calculate in-degrees for each node
        degrees = dict(topo.in_degree())
        degrees = torch.tensor(list(degrees.values()), dtype=torch.float32)
        degrees = degrees.reshape(-1, 1)
        
        # Avoid zero capacity values by setting a minimum
        mask = torch.where(capacities == 0)
        capacities[mask] += (1e-2 / self.scale) 

        # Get edge index to calculate capacity sums per node
        edge_index = self.get_edge_index()
        nodes = {node: i for i, node in enumerate(topo.nodes())}

        # Calculate the sum of capacities for edges connected to each node
        cap_sum_list = []
        for node in topo.nodes():
            node_id = nodes[node]
            indices = (edge_index[0] == node_id).nonzero()
            cap_sum = torch.sum(self.capacities[indices])
            cap_sum_list.append(cap_sum)

        cap_sum_list = torch.tensor(cap_sum_list)
        cap_sum_list = cap_sum_list.reshape(-1, 1)
        # Combine degrees and capacity sums as node features
        node_features = torch.cat((degrees, cap_sum_list), dim=1)
        
        self.node_features = node_features
        return node_features
    
    def read_opts_from_txt(self, opts_filename):
        """ Read optimal values from a txt file.

        Args:
            opts_filename (str): path to the txt file containing the optimal values
        
        Returns:
            numpy.array: optimal values
        """

        try:
           file = open(opts_filename, "r")
           opts = np.loadtxt(file).ravel()
           file.close()
        except:
            raise FileNotFoundError(f"File {opts_filename} not found. Please run the pre-processing script first.")
        
        return opts
    
    def sample_silding_windows(self, tms, window_size):
        """
        Create a sliding window of traffic matrices.

        Args:
            tms (numpy.array): traffic matrices
            window_size (int): size of the sliding window

        Returns:
            list: list of sliding windows
        """

        # Generate sequences of traffic matrices of specified window size
        windows = []
        for i in range(len(tms) - window_size):
            window = tms[i:i + window_size]
            windows.append(window)

        return windows

    def get_edge_index(self):
        """ Get the edge index of the graph."""
        
        # Return cached result if already computed
        if self.edge_index is not None:
            return self.edge_index

        # Create mapping from node names to indices
        nodes = {node: i for i, node in enumerate(self.topo.nodes())}

        source_nodes = []
        target_nodes = []

        # Build edge index by converting edges to source-target pairs
        for i, j in self.topo.edges():
                x = nodes[i]
                y = nodes[j]
                source_nodes.append(x)
                target_nodes.append(y)
                
        # Create tensor with source and target node indices
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.int64)
        self.edge_index = edge_index

        return edge_index
    
    def find_mandatory_edges(self, pij):
        """
        Find all edges that must be included in every path between some source-destination pairs.

        Args:
            pij (dict): Dictionary where each key is a (src, dst) pair, and the value is a list of paths.
                        Each path is represented as a list of edge tuples.

        Returns:
            Set of edge tuples that are mandatory for at least one OD (origin-destination) pair.
        """ 
        mandatory_edges = set()

        # Return cached result if already computed
        if self.mandatory_edges is not None:
            return self.mandatory_edges

        # Process each source-destination pair to find mandatory edges
        for (src, dst), paths in pij.items():
            if not paths:
                continue  # Skip if no paths are provided for this pair

            # Convert each path to a set of edges
            path_edge_sets = [set(path) for path in paths]
        
            # Find the intersection of all paths: edges that appear in every path
            common_edges = set.intersection(*path_edge_sets)
        
            # Add those edges to the final result
            mandatory_edges.update(common_edges)

        self.mandatory_edges = mandatory_edges
        return mandatory_edges

    # Dataset length method - returns number of samples in the dataset
    def __len__(self):
        return self.tm_seqences.shape[0]
    
    # Dataset getitem method - returns a specific sample from the dataset
    def __getitem__(self, index):
        """ Get item from the dataset. """
        return self.tm_seqences[index], self.tm_preds[index]
    
    # Normalize input tensor by scale factor
    def normalize(self, x): 
        """ Normalize the input tensor. """
        return x / self.scale

    # Denormalize input tensor by scale factor
    def denormalize(self, x): 
        """ Denormalize the input tensor. """
        return x * self.scale
    
    # Get standard deviation of traffic matrices for training set
    def get_tm_histories_std(self):
        """Get the standard deviation of the train traffic per s-d pairs."""
        tm_hist_std = np.std(self.train_tms, axis=0)
        return tm_hist_std
