import gurobipy as grb

from .useful_functions import Get_path_capacities


class LinearPragramming(object):
    """
    A class to solve traffic engineering problems using linear programming techniques.
    This class implements optimal routing algorithms that minimize maximum link utilization (MLU).
    """
    
    def __init__(self, topology, candidate_path, edge_to_path):
        """
        Initialize the LinearPragramming instance with network topology and path information.
        
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

    def MLU_traffic_engineering(self, demands):
        """
        Compute the traffic engineering solutions for multiple demands to minimize the worst-case MLU (Maximum Link Utilization).
        This method finds optimal routing weights to minimize the maximum utilization across all links in the network.

        Args:
            demands: The traffic demands, shape: (demand_number, number_of_nodes * number_of_nodes)
                     Each demand represents traffic flow between node pairs

        Returns:
            tuple: (optimal_MLU_value, path_weight_routing)
                   optimal_MLU_value: The minimized maximum link utilization value
                   path_weight_routing: Dictionary mapping path identifiers to their routing weights
        """
        # Reshape each demand matrix from 1D to 2D (nodes x nodes)
        for i in range(len(demands)):
            demands[i] = demands[i].reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
            
        # Create Gurobi optimization model
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0  # Disable Gurobi output for cleaner execution
        
        # Decision variable: MLU to be minimized
        mlu = m.addVar(lb = 0, vtype = grb.GRB.CONTINUOUS, name = 'mlu')
        
        # Generate variable names for path weights in format 'w_i_j_k'
        # where i,j are source-destination nodes and k is the path index
        name_path_weight = [f'w_{i}_{j}_{k}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i  # Exclude self-loops (i == j)
                    for k in range(len(self.candidate_path[(i, j)]))
                    ]
        # Add path weight variables (continuous between 0 and 1)
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

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
        
        # For each demand matrix, add flow constraints on each edge
        # The total flow on an edge (sum of traffic from all paths using this edge) 
        # must not exceed the edge's capacity multiplied by the MLU factor
        for demand in demands:
            m.addConstrs(
                grb.quicksum(
                    path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                ) <= mlu * self.topology.edges[edge]['capacity']
                for edge in self.topology.edges
            )
        
        # Objective: Minimize the MLU (maximum link utilization)
        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        
        # Process the solution if optimization was successful
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution') 

    def Spread_traffic_engineering(self, demand, Spread):
        """
        Compute the traffic engineering solutions for a single demand to minimize the worst-case MLU 
        with Google's hedging mechanism. This method incorporates a spread factor that constrains
        how much traffic can be routed along different paths based on path capacity limits.
        
        Args:
            demand: The traffic demand, shape: (number_of_nodes * number_of_nodes)
                    Represents traffic flow between node pairs
            Spread: The spread factor that controls how traffic is distributed across paths

        Returns:
            tuple: (optimal_MLU_value, path_weight_routing)
                   optimal_MLU_value: The minimized maximum link utilization value
                   path_weight_routing: Dictionary mapping path identifiers to their routing weights
        """
        # Reshape demand matrix from 1D to 2D (nodes x nodes) if needed
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        else:
            demand = demand
            
        # Create Gurobi optimization model
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0

        # Decision variable: MLU to be minimized
        mlu = m.addVar(lb = 0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        
        # Generate variable names for path weights in format 'w_i_j_k'
        name_path_weight = [f'w_{i}_{j}_{k}'
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i  # Exclude self-loops (i == j)
            for k in range(len(self.candidate_path[(i, j)]))
            ]
        # Add path weight variables (continuous between 0 and 1)
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')
        
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
        
        # Flow constraint: Total traffic on each edge must not exceed capacity * MLU
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
            ) <= mlu * self.topology.edges[edge]['capacity']
            for edge in self.topology.edges
        )
        
        # Google's hedging constraint: Limits how traffic is distributed based on path capacity
        # This constraint ensures path weights are proportional to path capacity with a spread factor
        m.addConstrs(
            path_weight[f'w_{i}_{j}_{k}'] * (Spread * grb.quicksum(
                    self.path_capacity[(i, j, k)] for k in range(len(self.candidate_path[(i, j)]))
                )) <=  
                self.path_capacity[(i, j, k)] 
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i  # Exclude self-loops
            for k in range(len(self.candidate_path[(i, j)]))
        )
        
        # Objective: Minimize the MLU (maximum link utilization)
        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        
        # Process the solution if optimization was successful
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')
