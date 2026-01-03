import gurobipy as grb

from .useful_functions import Get_path_capacities


class Oblivious(object):
    """
    A class to solve oblivious traffic engineering problems using a dual formulation approach.
    This implements the method described in 'Making Intra-Domain Routing Robust to Changing Traffic',
    which finds oblivious routing weights that are robust to traffic demand changes.
    """
    
    def __init__(self, topology, candidate_path, edge_to_path):
        """
        Initialize the Oblivious instance with network topology and path information.
        
        Args:
            topology: NetworkX graph object representing the network topology with edge capacities
            candidate_path: Dictionary containing candidate paths between node pairs
            edge_to_path: Mapping from each edge to the paths that use that edge
        """
        self.topology = topology
        self.candidate_path = candidate_path
        # Calculate path capacities based on minimum edge capacity in each path
        self.path_capacity = Get_path_capacities(topology, candidate_path)
        self.edge_to_path = edge_to_path

    def solve_traffic_engineering(self):
        """
        Solve the oblivious traffic engineering problem using a dual formulation approach.
        
        This method implements the algorithm from 'Making Intra-Domain Routing Robust to Changing Traffic'
        which formulates oblivious routing as a dual linear program to minimize the oblivious ratio.
        The oblivious ratio measures how much worse the oblivious routing performs compared 
        to optimal traffic engineering for any given demand matrix.
        
        Returns:
            tuple: (optimal_oblivious_ratio, path_weight_solution)
                   optimal_oblivious_ratio: The minimized oblivious ratio value
                   path_weight_solution: Dictionary mapping path identifiers to their routing weights
        """
        # Create Gurobi optimization model for the dual formulation
        m = grb.Model('dual_traffic_engineering_grb')
        m.Params.OutputFlag = 0  # Disable Gurobi output for cleaner execution
        
        # Decision variable: oblivious ratio to be minimized
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

        # First constraint from the paper: limits the total capacity-weighted sum of pi variables
        # This constraint ensures the solution respects the network capacity
        for l in range(self.topology.number_of_edges()):
            m.addConstr(
                grb.quicksum(self.topology.edges[edge_list[j]]['capacity'] * pi[f'pi_{l}_{j}'] for j in range(self.topology.number_of_edges())) <= ratio
            )
                
        # Second constraint from the paper: relates flow fractions to p variables
        # This ensures that flow fractions don't exceed what's allowed by the dual variables
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if i!=j:  # Only for non-self-loop pairs
                        if f'f_{l}_{i}_{j}' in f_dict:  # Only if this flow exists
                            m.addConstr(
                                f_dict[f'f_{l}_{i}_{j}'] <= p[f'p_{i}_{j}_{l}'] * self.topology.edges[edge_list[l]]['capacity']
                            )
        
        # Third constraint from the paper: triangle inequality for shortest paths
        # This constraint ensures the dual variables satisfy a triangle inequality
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for e in range(self.topology.number_of_edges()):
                    # edge_list[e][0] is the src node of edge
                    # edge_list[e][1] is the dst node of edge
                    m.addConstr(
                        pi[f'pi_{l}_{e}'] + p[f'p_{i}_{edge_list[e][0]}_{l}'] - p[f'p_{i}_{edge_list[e][1]}_{l}'] >= 0
                    )
        
        # Fifth constraint from the paper: zero flow from a node to itself
        # This ensures no flow is assigned for self-loop cases
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                m.addConstr(p[f'p_{i}_{i}_{l}'] == 0)

        # Fourth and sixth constraints are handled by the lower bound (lb=0) in addVars.
        
        # Objective: Minimize the oblivious ratio
        m.setObjective(ratio, grb.GRB.MINIMIZE)
        m.optimize()
        
        # Process the solution if optimization was successful
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            return m.objVal, solution
        else:
            print('No solution') 