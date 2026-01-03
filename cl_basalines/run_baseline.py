import os
import torch
import torch
import argparse
import numpy as np

from cope import COPE
from oblivious_routing import Oblivious
from optimal_routing import LinearPragramming

from tqdm import tqdm 
from src.objective import compute_mlus
from src.utils.set_seed import set_seed
from src.build_dataloader import build_dataloader
from .predict_tools import weighted_moving_average_predict
from .useful_functions import Get_edge_to_path, get_weight_dict_tensor, expand_tms
from src.utils.read_data import read_paths_from_file, read_graph_from_json
from src.utils.useful_functions import compute_ksp_paths, get_paths_to_edges, get_commodities_to_paths


def parse_args(): 
    """
    Parse command line arguments for the classical traffic engineering script.
    
    Returns:
        argparse.Namespace: Parsed arguments including method, topology, file paths, and hyperparameters
    """
    parser = argparse.ArgumentParser(description='Classical TE Script') 
    parser.add_argument('--seed', type=int, default=2025, help='random seed') 
    parser.add_argument('--num_itrs', type=int, default=1, help='experiments times')

    # Method selection
    parser.add_argument('--method', type=str, default='optimal', 
                        choices=['cope', 'predte', 'oblivious', 'optimal'], 
                        help='method name, options: [cope, predte, oblivious, optimal]')
    
    # Topology configuration
    parser.add_argument("--topology", type=str, default='Abilene', 
                        choices=['Abilene', 'GEANT', 'CERNET', 'UsCarrier', 'Cogentco'], 
                        help="Name of the topology to be used.")
    
    # File path configuration
    parser.add_argument("--topology_filepath", type=str, default='./data/Abilene/topology.json', 
                        help="Name of .json file the topology was stored.")
    parser.add_argument("--tm_filepath", type=str, default='./data/Abilene/Abilene.csv', 
                        help="Name of .csv file the traffic matrices were stored.")
    parser.add_argument('--result_path', type=str, default='./results/', 
                        help='location of computed mlus')
    
    # Path and window configuration
    parser.add_argument("--num_paths", type=int, default=8, 
                        help="Number of optimized tunnels per OD pair for searching.")
    parser.add_argument('--window_size', type=int, default=12, 
                        help='history traffic matrix sequence length') 
    parser.add_argument("--scale", type=int, default=10**9, help="Normalized scale.")

    # Batch size configuration
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size of model evaluation')

    # COPE-specific parameters
    parser.add_argument('--beta', type=float, default = 1.5, help='useful factor in COPE') 
    parser.add_argument('--num_preds', type=float, default = 100, help='size of prediction set in COPE') 

    args = parser.parse_args() 
    return args 


if __name__ == '__main__': 
    # Parse command line arguments
    args = parse_args() 
    set_seed(args.seed) 

    # Read topology and capacities from JSON file
    topo, capacities = read_graph_from_json(args.topology_filepath) 
    num_nodes = len(topo.nodes()) 

    # Load or compute K-shortest paths for traffic engineering
    try:
        te_paths = read_paths_from_file(filepath=f'./data/{args.topology}/paths.txt', num_nodes=num_nodes)
    except: 
        # Generate paths if they don't exist
        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j] 
        te_paths = compute_ksp_paths(k=args.num_paths, pairs=pairs, graph=topo, save2txt=True, 
                                     filepath=f'./data/{args.topology}', transform=True) 
    
    # Load paths in converted format for path-to-edge matrix computation
    paths = read_paths_from_file(filepath=f'./data/{args.topology}/paths.txt', num_nodes=num_nodes, convert=True)
    
    # Create path-to-edges matrix as a sparse tensor
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

    # Run experiments for specified number of iterations
    for ii in range(args.num_itrs): 
        setting = '{}_{}'.format(args.topology, args.method) 

        # Build data loaders for training, validation and testing
        _, _, test_loader = build_dataloader(args.topology_filepath, args.tm_filepath, args.batch_size, 
                                             args.scale, args.eval_batch_size, args.window_size, 
                                             split_ratio=(0.7, 0.1, 0.2)) 
        
        results = [] 
        split_ratios = [] 
        histories = test_loader.dataset.tm_seqences.float() 
        targets = test_loader.dataset.tm_preds.float() 

        # Map edges to paths for traffic engineering calculations
        edge_to_path = Get_edge_to_path(topo, te_paths) 

        # Apply the selected traffic engineering method
        if args.method == 'predte': 
            # Predict traffic using weighted moving average and solve with linear programming
            preds = weighted_moving_average_predict(histories.cpu().numpy()) * args.scale
            preds = expand_tms(preds, num_nodes) 
            te_scheme = LinearPragramming(topo, te_paths, edge_to_path) 

            for i in tqdm(range(len(test_loader.dataset)), total=len(test_loader.dataset)): 
                _, split_ratios_dict = te_scheme.MLU_traffic_engineering(demands=[preds[i, :]]) 
                split_ratio = get_weight_dict_tensor(split_ratios_dict, num_nodes, args.num_paths) 
                split_ratios.append(split_ratio) 

            split_ratios = torch.stack(split_ratios, dim=0)

        elif args.method == 'oblivious': 
            # Apply oblivious routing method
            te_scheme = Oblivious(topo, te_paths, edge_to_path) 
            _, split_ratios_dict = te_scheme.solve_traffic_engineering()
             
            worse_bound_split_ratio = get_weight_dict_tensor(split_ratios_dict, num_nodes, args.num_paths) 
            split_ratios = worse_bound_split_ratio.unsqueeze(0).repeat(len(test_loader.dataset), 1)

        elif args.method == 'cope': 
            # Apply COPE (Capacity Oriented Path Enforcement) method
            te_scheme = COPE(topo, te_paths, edge_to_path) 
            preds = weighted_moving_average_predict(histories.cpu().numpy()) * args.scale
            preds = expand_tms(preds, num_nodes)[torch.randperm(preds.shape[0])[:args.num_preds]] 
            predict_dms_list = [preds[i] for i in range(args.num_preds)]
            _, split_ratios_dict = te_scheme.solve_traffic_engineering(args.beta, predict_dms_list)
            bounded_split_ratio = get_weight_dict_tensor(split_ratios_dict, num_nodes, args.num_paths) 
            split_ratios = bounded_split_ratio.unsqueeze(0).repeat(len(test_loader.dataset), 1)
        
        else: 
            # Apply optimal routing using linear programming with actual demands
            te_scheme = LinearPragramming(topo, te_paths, edge_to_path) 
            mlus = []

            for i, tm in tqdm(enumerate(expand_tms(test_loader.dataset.tm_preds.cpu().numpy(), num_nodes) * args.scale), total=len(test_loader.dataset)): 
                mlu, split_ratios_dict = te_scheme.MLU_traffic_engineering(demands=[tm]) 
                split_ratio = get_weight_dict_tensor(split_ratios_dict, num_nodes, args.num_paths) 
                split_ratios.append(split_ratio) 
                mlus.append(mlu)

            split_ratios = torch.stack(split_ratios, dim=0) 

        # Compute MLU (Maximum Link Utilization) results
        results_ii = compute_mlus(split_ratios, targets, topo, paths_to_edges, commodities_to_paths, scale=args.scale) 
        
        print('Average MLU: ', np.average(results_ii)) 
        results.append(results_ii) 

    # Save results to specified directory
    results_save_path = os.path.join(args.result_path, setting) 
    if not os.path.exists(results_save_path): 
        os.makedirs(results_save_path) 

    # np.save(results_save_path + '/mlu_results.npy', np.array(results))
