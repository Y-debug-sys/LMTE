import gurobipy as grb

from .oblivious_routing import Oblivious


class COPE(object):
    """
    Combined Oblivious and Predictive Engineering (COPE) class.
    This class implements a hybrid traffic engineering approach that combines oblivious routing
    (robust to traffic uncertainty) with predictive routing (optimized for predicted demands).
    The algorithm balances between worst-case performance and performance on predicted demands
    using a beta parameter that controls the trade-off between these two approaches.
    """
    
    def __init__(self, topology, candidate_path, edge_to_path):
        """
        Initialize the COPE instance with network topology and path information.
        
        Args:
            topology: NetworkX graph object representing the network topology with edge capacities
            candidate_path: Dictionary containing candidate paths between node pairs
            edge_to_path: Mapping from each edge to the paths that use that edge
        """
        self.topology = topology
        self.candidate_path = candidate_path
        self.edge_to_path = edge_to_path
        # Create an Oblivious instance to calculate oblivious routing ratio
        self.oblivious = Oblivious(topology, candidate_path, edge_to_path)

    def solve_traffic_engineering(self, beta, predict_dms):
        """
        Solve the COPE traffic engineering problem that balances oblivious and predictive routing.
        
        This method implements a hybrid approach that optimizes for both predicted demands
        and worst-case scenarios. It uses a penalty ratio based on the oblivious ratio scaled
        by beta to ensure robustness while optimizing for predicted traffic demands.
        
        Args:
            beta (float): A parameter that controls the trade-off between oblivious and predictive routing.
                         Higher values of beta increase the importance of robustness to traffic uncertainty.
            predict_dms (list): List of predicted demand matrices, each with shape (num_nodes, num_nodes)
                               representing traffic demands between node pairs
        
        Returns:
            tuple: (optimal_ratio, path_weight_solution)
                   optimal_ratio: The minimized ratio value for the hybrid approach
                   path_weight_solution: Dictionary mapping path identifiers to their routing weights
        """
        # Reshape each predicted demand matrix from 1D to 2D (nodes x nodes)
        for idx, demand in enumerate(predict_dms):
            predict_dms[idx] = demand.reshape(self.topology.number_of_nodes(), self.topology.number_of_nodes())
            
        # Calculate the oblivious ratio using the Oblivious class
        oblivious_ratio, _ = self.oblivious.solve_traffic_engineering()
        # Calculate the penalty ratio as beta times the oblivious ratio
        plenalty_ratio = beta * oblivious_ratio
        
        # Create Gurobi optimization model for the COPE formulation
        m = grb.Model('dual_traffic_engineering_grb')
        m.Params.OutputFlag = 0  # Disable Gurobi output for cleaner execution
        
        # Decision variable: ratio to be minimized (the maximum link utilization for predicted demands)
        ratio = m.addVar(lb = 0, vtype=grb.GRB.CONTINUOUS, name='obli_ratio')
        
        # Data structures for mapping edges to indices and tracking edge-path relationships
        edge_dict = {}
        edge_list = []
        
        # Create a list of edges for consistent indexing
        for edge in self.topology.edges:
            edge_list.append(edge)

        # Create mapping from (edge_idx, src, dst) to path indices that use this edge
        # This helps efficiently compute the fraction of flow from each (src, dst) pair on each edge
        edge_src_dst_to_k = {} 
        for l in range(self.topology.number_of_edges()):
            for (src,dst,k) in self.edge_to_path[edge_list[l]]:
                if (l,src,dst) not in edge_src_dst_to_k:
                    edge_src_dst_to_k[(l,src,dst)] = [k]
                else:
                    edge_src_dst_to_k[(l,src,dst)].append(k)

        # Map each edge to its index for consistent referencing
        for i in range(self.topology.number_of_edges()):
            edge_dict[edge_list[i]] = i

        # Decision variables: path weights (routing fractions for each path)
        # Format: 'w_i_j_k' where i,j are source-destination nodes and k is the path index
        name_path_weight = [f'w_{i}_{j}_{k}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i  # Exclude self-loops (i == j)
                    for k in range(len(self.candidate_path[(i, j)]))
                    ]
        # Add path weight variables (continuous between 0 and 1)
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        # Decision variables: fraction of flow from each (src, dst) pair on each edge
        # Format: 'f_l_i_j' where l is the edge index and i,j are source-destination nodes
        name_f_dict = [f'f_{l}_{i}_{j}'
                       for l in range(self.topology.number_of_edges())
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i  # Exclude self-loops (i == j)
                    ]
        f_dict = m.addVars(name_f_dict, lb=0, vtype=grb.GRB.CONTINUOUS, name='f_dict')

        # Decision variables: pi (dual variables related to capacity constraints)
        # Format: 'pi_i_j' where i,j are edge indices
        name_pi = [f'pi_{i}_{j}'
                    for i in range(self.topology.number_of_edges())
                    for j in range(self.topology.number_of_edges())]
        pi = m.addVars(name_pi, lb=0, vtype=grb.GRB.CONTINUOUS, name='pi')

        # Decision variables: p (dual variables related to shortest path distances)
        # Format: 'p_i_j_l' where i,j are node indices and l is the edge index
        name_p = [f'p_{i}_{j}_{l}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    for l in range(self.topology.number_of_edges())]
        p = m.addVars(name_p, lb=0, vtype=grb.GRB.CONTINUOUS, name='p')

        # Constraint: For each source-destination pair, the sum of routing weights of all candidate paths
        # must equal 1 (all traffic must be routed through some path)
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i  # Exclude self-loops
        )
        
        # Constraint: Calculate the fraction of flow from each (src, dst) pair on each edge
        # This links the path weights to the flow fractions on edges
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if (l,i,j) in edge_src_dst_to_k.keys():
                        # The flow fraction on edge l from src i to dst j is the sum of weights
                        # of all paths from i to j that use edge l
                        m.addConstr(
                            f_dict[f'f_{l}_{i}_{j}'] == grb.quicksum(path_weight[f'w_{i}_{j}_{k}'] for k in edge_src_dst_to_k[(l,i,j)])
                        )

        # First constraint from the oblivious formulation: limits the total capacity-weighted sum of pi variables
        # Uses the penalty ratio instead of the oblivious ratio to balance robustness and performance
        for l in range(self.topology.number_of_edges()):
            m.addConstr(
                grb.quicksum(self.topology.edges[edge_list[j]]['capacity'] * pi[f'pi_{l}_{j}'] for j in range(self.topology.number_of_edges())) <= plenalty_ratio
            )
                
        # Second constraint from the oblivious formulation: relates flow fractions to p variables
        # This ensures that flow fractions don't exceed what's allowed by the dual variables
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if i!=j:  # Only for non-self-loop pairs
                        if f'f_{l}_{i}_{j}' in f_dict:  # Only if this flow exists
                            m.addConstr(
                                f_dict[f'f_{l}_{i}_{j}'] <= p[f'p_{i}_{j}_{l}'] * self.topology.edges[edge_list[l]]['capacity']
                            )
        
        # Third constraint from the oblivious formulation: triangle inequality for shortest paths
        # This constraint ensures the dual variables satisfy a triangle inequality
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for e in range(self.topology.number_of_edges()):
                    # edge_list[e][0] is the src node of edge
                    # edge_list[e][1] is the dst node of edge
                    m.addConstr(
                        pi[f'pi_{l}_{e}'] + p[f'p_{i}_{edge_list[e][0]}_{l}'] - p[f'p_{i}_{edge_list[e][1]}_{l}'] >= 0
                    )
        
        # Fifth constraint from the oblivious formulation: zero flow from a node to itself
        # This ensures no flow is assigned for self-loop cases
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                m.addConstr(p[f'p_{i}_{i}_{l}'] == 0)

        # Additional constraint specific to COPE: for each predicted demand matrix,
        # ensure that the total flow on each edge doesn't exceed the ratio times edge capacity
        # This constraint optimizes the routing for the predicted demands
        for demand in predict_dms:
            m.addConstrs(
                grb.quicksum(
                    path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                ) <= ratio * self.topology.edges[edge]['capacity']
                for edge in self.topology.edges
            )
        
        # Objective: Minimize the ratio (maximum link utilization for predicted demands)
        m.setObjective(ratio, grb.GRB.MINIMIZE)
        m.optimize()
        
        # Process the solution if optimization was successful
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            return m.objVal, solution
        else:
            print('No solution')