import os
import torch
import argparse
import numpy as np

from torch import optim
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs

from src.model import LmteModel
from src.solver import LmteSolver
from src.utils.set_seed import set_seed
from src.build_dataloader import build_dataloader
from src.utils.read_data import read_paths_from_file, read_graph_from_json
from src.utils.useful_functions import compute_ksp_paths, get_paths_to_edges, get_commodities_to_paths


def parse_args(): 
    """
    Parse command-line arguments for the traffic engineering model.
    
    Returns:
        Parsed arguments object containing all configuration parameters
    """
    # Initialize argument parser with description
    parser = argparse.ArgumentParser(description='Main Script') 
    parser.add_argument('--seed', type=int, default=2025, help='random seed') 

    # Topology-related arguments
    parser.add_argument("--topology", type=str, default='GEANT', 
                        choices=['Abilene', 'GEANT', 'CERNET', 'UsCarrier', 'Cogentco'], 
                        help="Name of the topology to be used.")
    parser.add_argument("--topology_filepath", type=str, default='./data/GEANT/topology.json', 
                        help="Name of .json file the topology was stored.")
    parser.add_argument("--tm_filepath", type=str, default='./data/GEANT/GEANT.csv', 
                        help="Name of .csv file the traffic matrices were stored.")
    
    # Training-related arguments
    parser.add_argument('--num_itrs', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    
    # Experiment status and configuration arguments
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument("--num_paths", type=int, default=8, 
                        help="Number of optimized tunnels per OD pair for searching.")
    parser.add_argument('--window_size', type=int, default=12, 
                        help='history traffic matrix sequence length')
    parser.add_argument("--scale", type=int, default=10**9, help="Normalized scale.")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', 
                        help='location of model checkpoints') 
    parser.add_argument('--result_path', type=str, default='./results/', 
                        help='location of computed mlus')

    # Model architecture arguments
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, default=1, help='LLM model layers')
    parser.add_argument('--d_keys', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--num_gnn_layers', type=int, default=3, help='num of GNN layers')
    parser.add_argument('--num_rnn_layers', type=int, default=2, help='num of RNN layers')
    parser.add_argument('--num_dnn_layers', type=int, default=4, help='num of Dense layers')
    parser.add_argument('--dropout', type=float, default=0., help='dropout')
    parser.add_argument('--llm_model', type=str, default='llama-8b',
                        help='model name, options: [llama-8b, llama-3b, llama-1b, llama-13b]')
    parser.add_argument('--is-lightweight', dest="is_divide", action='store_true',
                        help='whether to use route-aware implementation') 
    
    # Failure and burst testing arguments
    parser.add_argument('--add_failures', type=int, default=0,
                        help='whether to add failures during testing')
    parser.add_argument('--num_failures', type=int, default=1, 
                        help='number of failures to add during testing')
    parser.add_argument('--add_bursts', type=int, default=0, 
                        help='whether to add bursts during testing')
    parser.add_argument('--burst_factor', type=float, default=10., 
                        help='burst factor for traffic matrix fluctuations')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__': 
    """
    Main execution function for the traffic engineering model.
    This function handles:
    1. Parsing command-line arguments
    2. Setting up the training environment with Accelerate
    3. Loading topology and traffic matrix data
    4. Computing or loading path information
    5. Setting up model, optimizer, and solver
    6. Training and testing the model
    7. Saving the results
    """
    args = parse_args()
    set_seed(args.seed)  # Set random seed for reproducibility

    # Configure distributed training with Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # DDP config
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./deepspeed_cfg.json')  # DeepSpeed config
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)  # Accelerator config

    # Load network topology and capacities from file
    topo, capacities = read_graph_from_json(args.topology_filepath) 
    num_nodes = len(topo.nodes())  # Number of nodes in the network
    num_edges = len(topo.edges())  # Number of edges in the network
    num_paths = args.num_paths  # Number of paths per OD pair

    # Compute or load tunnel paths for the network
    tunnels_filename = f'./data/{args.topology}/paths.txt'
    if not os.path.exists(tunnels_filename): 
        # Generate k-shortest paths for all OD pairs if not already computed
        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        _ = compute_ksp_paths(k=args.num_paths, pairs=pairs, graph=topo, save2txt=True, 
                              filepath=f'./data/{args.topology}', transform=True)
        
    # Load the computed paths
    paths = read_paths_from_file(filepath=tunnels_filename, num_nodes=num_nodes, convert=True)

    # Create paths-to-edges matrix as a sparse tensor
    p_matrix = get_paths_to_edges(topo, paths=paths)
    pm_coo = p_matrix.tocoo()
    paths_to_edges = torch.sparse_coo_tensor(np.vstack((pm_coo.row, pm_coo.col)), \
                                             torch.FloatTensor(pm_coo.data), 
                                             torch.Size(pm_coo.shape)) 
    
    # Create commodities-to-paths matrix as a sparse tensor
    c_matrix = get_commodities_to_paths(topo, num_paths=p_matrix.shape[0], paths=paths)
    cm_coo = c_matrix.tocoo()
    commodities_to_paths = torch.sparse_coo_tensor(np.vstack((cm_coo.row, cm_coo.col)), \
                                                   torch.FloatTensor(cm_coo.data), 
                                                   torch.Size(cm_coo.shape)) 
    
    results = []  # Store results from multiple iterations

    # Run multiple iterations if specified
    for ii in range(args.num_itrs):
        setting = '{}_lmte'.format(args.topology)

        # Build data loaders for training, validation, and testing
        train_loader, valid_loader, test_loader = build_dataloader(args.topology_filepath, args.tm_filepath, args.batch_size, 
                                                                   args.scale, args.eval_batch_size, args.window_size, 
                                                                   split_ratio=(0.7, 0.1, 0.2))
        
        # Get the maximum path length for model configuration
        max_path_length = train_loader.dataset.get_padded_edge_ids_per_path(paths).shape[-1]
        
        # Initialize the LMTE model with specified parameters
        model = LmteModel(args.d_model, args.llm_dim, num_nodes, num_paths, num_edges, args.num_gnn_layers, args.num_rnn_layers, 
                          args.num_dnn_layers, args.d_keys, args.window_size, args.n_heads, args.dropout, d_middle=args.d_model, 
                          llm_layers=args.llm_layers, llm_model=args.llm_model, max_length=max_path_length, 
                          use_divide_head=args.is_divide).float() 
        
        # Collect parameters that require training
        trained_parameters = []
        for p in model.parameters(): 
            if p.requires_grad is True: 
                trained_parameters.append(p)

        # Initialize optimizer with model parameters and learning rate
        optimizer = optim.Adam(trained_parameters, lr=args.learning_rate)
        # Prepare all components for distributed training with Accelerate
        train_loader, valid_loader, test_loader, model, optimizer = accelerator.prepare(train_loader, valid_loader, test_loader, model, optimizer) 
        
        # Set up model save path for this iteration
        model_save_path = os.path.join(args.checkpoint_path, setting+f'_{ii}')
        if not os.path.exists(model_save_path): 
            os.makedirs(model_save_path)  # Create directory if it doesn't exist
            
        # Initialize the solver with all required components
        solver = LmteSolver(args, model, topo, accelerator, train_loader, valid_loader, commodities_to_paths, paths_to_edges,
                            optimizer, paths) 
        
        if args.is_training: 
            # Train the model if in training mode
            solver.adapt(model_save_path, auto_fail=True, auto_burst=True) 
        else:
            # Load a pre-trained model if in testing mode only
            solver.load_checkpoint(model_save_path)

        # Test the model and collect results
        results_ii = solver.test(test_loader, args.add_failures, args.num_failures, args.add_bursts, args.burst_factor)
        
        results.append(results_ii)

    # Prepare directory and save results
    results_save_path = os.path.join(args.result_path, setting) 
    if not os.path.exists(results_save_path): 
        os.makedirs(results_save_path) 

    # Determine the appropriate filename based on whether failures or bursts were added
    if args.add_failures:
        save_filename = '/mlu_results_f.npy'
    elif args.add_bursts:
        save_filename = '/mlu_results_b.npy'
    else:
        save_filename = '/mlu_results.npy'
    
    # Save the results as a numpy array
    np.save(results_save_path + save_filename, np.array(results))