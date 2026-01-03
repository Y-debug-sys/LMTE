import os
import time
import torch
import random
import numpy as np

from .objective import compute_mlus
from .utils.early_stopping import EarlyStopping
from .utils.useful_functions import get_capacities_from_graph, mask_invalid_paths


class LmteSolver(object): 
    """
    Solver class for the LLM-based Traffic Engineering (LMTE) model.
    Handles training, validation, and testing of the LMTE model.
    """
    
    def __init__(
        self, 
        args,
        model,
        topology,
        accelerator,
        train_loader,
        valid_loader,
        commodities_to_paths,
        paths_to_edges,
        optimizer,
        tunnels
    ): 
        """
        Initialize the LmteSolver.
        
        Args:
            args: Arguments containing model and training parameters
            model: The LMTE model to be trained
            topology: Network topology information
            accelerator: Accelerator for distributed training
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            commodities_to_paths: Matrix mapping commodities to paths
            paths_to_edges: Matrix mapping paths to edges
            optimizer: Optimizer for training
            tunnels: Tunnels information for the network
        """
        self.args = args 
        self.model = model
        self.topology = topology 
        self.optimizer = optimizer
        self.accelerator = accelerator 
        self.train_loader, self.valid_loader = train_loader, valid_loader 

        # Move matrices to the appropriate device
        self.paths_to_edges = paths_to_edges.to(accelerator.device) 
        self.commodities_to_paths = commodities_to_paths.to(accelerator.device) 
        # Initialize early stopping with patience and verbose settings
        self.early_stopping = EarlyStopping(accelerator, args.patience, verbose=True) 

        # Initialize network structure data and move to device
        self.edge_index = train_loader.dataset.get_edge_index().to(accelerator.device)
        self.node_features = train_loader.dataset.get_node_features().float().to(accelerator.device)
        self.capacities = train_loader.dataset.capacities.float().to(accelerator.device)
        self.edge_ids_per_path = train_loader.dataset.get_padded_edge_ids_per_path(tunnels).to(accelerator.device)
        self.mandatory_edges = train_loader.dataset.find_mandatory_edges(tunnels)

    def adapt(self, save_path=None, auto_fail=True, auto_burst=True): 
        """
        Adapt (train) the LMTE model.
        
        The function performs the following steps:
        1. Initialize the early stopping object.
        2. Create the checkpoint directory.
        3. Loop over the epochs.
        4. Loop over the training dataloader.
        5. Compute the loss.
        6. Backward pass.
        7. Step the optimizer.
        8. Loop over the validation dataloader.
        9. Compute the validation loss.
        10. Print the losses.
        11. Check for early stopping.
        12. If early stopping, break the loop.
        13. Load the best model.
        14. Save the model.
        
        Args:
            save_path: Path to save model checkpoints
            auto_fail: Whether to automatically inject link failures during training
            auto_burst: Whether to automatically inject traffic bursts during training
        """
        save_path = save_path if save_path else self.args.checkpoint_path
        # Create save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            train_losses, valid_losses = [], []
            train_steps = len(self.train_loader)

            # Training loop
            for i, (tms, preds) in enumerate(self.train_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                # Move data to device
                tms = tms.float().to(self.accelerator.device) 
                preds = preds.float().to(self.accelerator.device) 

                bursty_tensor = None
                topology = self.topology.copy()
                capacities = self.capacities
                node_features = self.node_features 

                # Inject link failures with 10% probability
                if auto_fail: 
                    r = torch.randint(0, 10, (1,)).item() / 10 
                    if r < 0.1: 
                        node_features, topology, capacities = self.add_failures() 

                # Inject traffic bursts with 10% probability
                if auto_burst: 
                    r = torch.randint(0, 10, (1,)).item() / 10 
                    if r < 0.1: 
                        tms, preds = self.add_bursts(tms, preds) 
                        bursty_tensor = torch.ones((tms.shape[0],), device=self.accelerator.device) 

                split_ratios = self.model(tms, node_features, self.edge_index, capacities,
                                          topology, bursty_tensor, self.edge_ids_per_path) 

                loss = self.loss(split_ratios, preds, topology) 
                train_losses.append(loss.item()) 

                if (i + 1) % 50 == 0: 
                    self.accelerator.print( 
                        "\titers: {0}, epoch: {1} | normalized loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if not torch.isnan(loss):
                   self.accelerator.backward(loss)
                   self.optimizer.step()
            
            self.accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)) 

            self.model.eval()
            with torch.no_grad():

                for i, (tms, preds) in enumerate(self.valid_loader): 
                    tms = tms.float().to(self.accelerator.device)
                    preds = preds.float().to(self.accelerator.device)

                    split_ratios = self.model(tms, self.node_features, self.edge_index, self.capacities,
                                              self.topology, None, self.edge_ids_per_path)

                    mlus = compute_mlus(split_ratios, preds, self.topology, self.paths_to_edges, self.commodities_to_paths,
                                       scale=self.valid_loader.dataset.scale)
                    valid_losses += mlus

            valid_loss = np.average(valid_losses)
            train_loss = np.average(train_losses)
            self.accelerator.print("Epoch: {0} | Normalized Train Loss: {1:.7f}, Valid Loss: {2:.7f}".format(epoch + 1, train_loss, valid_loss))
        
            self.early_stopping(valid_loss, self.model, save_path)
            if self.early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break
        
        self.accelerator.wait_for_everyone()
            
    @torch.no_grad()
    def test(
        self, 
        test_loader, 
        add_failures=False, 
        num_failures=1, 
        add_bursts=False, 
        burst_factor=10.
    ): 
        """
        Test the LMTE model on a given test dataset.
        
        Args:
            test_loader: DataLoader for test data
            add_failures: Whether to inject link failures during testing
            num_failures: Number of link failures to inject
            add_bursts: Whether to inject traffic bursts during testing
            burst_factor: Factor by which to scale traffic bursts
        
        Returns:
            List of MLU (Maximum Link Utilization) values for each test sample
        """
        self.model.eval()
        results = [] 
        for i, (tms, preds) in enumerate(test_loader): 
            tms = tms.float().to(self.accelerator.device)
            preds = preds.float().to(self.accelerator.device) 

            if add_failures: 
                node_features, topology, capacities = self.add_failures(num_failures) 
            else:
                topology = self.topology
                capacities = self.capacities
                node_features = self.node_features 

            if add_bursts:
                tms, preds = self.add_bursts(tms, preds, burst_factor) 
                bursty_tensor = torch.ones((tms.shape[0],), device=self.accelerator.device) 
            else:
                bursty_tensor = None 
            
            split_ratios = self.model(tms, node_features, self.edge_index, capacities,
                                      topology, bursty_tensor, self.edge_ids_per_path) 

            result = self.loss(split_ratios, preds, topology, False) 
            results += result

        self.accelerator.print('Average MLU: ', np.average(results)) 
        return results 

    def add_failures(self, num_failures=1): 
        """
        Inject link failures into the network topology.
        
        Args:
            num_failures: Number of link failures to inject
        
        Returns:
            Updated node features, topology, and capacities
        """
        topology = self.topology.copy()
        edges = list(topology.edges())
        # failed_links = random.sample(edges, num_failures)

        for _ in range(1000):
            failed_links = random.sample(edges, num_failures)

            # Check if any failed link is a mandatory edge (either direction)
            if not any((e in self.mandatory_edges or (e[1], e[0]) in self.mandatory_edges) for e in failed_links):
                break

        for u, v in failed_links:
            topology[u][v]['capacity'] = 0

        capacities = [float(data['capacity']) for u, v, data in topology.edges(data=True)]
        node_features = self.train_loader.dataset.get_node_features().float()
        capacities = self.train_loader.dataset.normalize(torch.tensor(capacities))
        return node_features.to(self.accelerator.device), topology, capacities.to(self.accelerator.device)

    def add_bursts(self, histories, preds, scale_factor=10): 
        """
        Inject traffic bursts into the traffic matrices.
        
        Args:
            histories: Traffic matrices
            preds: Predicted traffic matrices
            scale_factor: Factor by which to scale traffic bursts
        
        Returns:
            Updated traffic matrices and predictions
        """
        diff = histories[:, 1:, :] - histories[:, :-1, :] 
        variance = diff.var(dim=1, keepdim=True)
        stddev = torch.sqrt(variance * scale_factor)
        noise = torch.randn_like(histories) * stddev
        histories = torch.clamp(histories + noise, min=0.)
        # histories[:, -1, :] = histories[:, -1, :] * 2
        return histories, preds

    def loss(self, x, targets, topology, train=True): 
        """
        Compute the loss for the LMTE model.
        
        Args:
            x: Split ratios output by the model
            targets: Ground truth traffic matrices
            topology: Network topology information
            train: Whether the loss is being computed for training
        
        Returns:
            Loss value(s)
        """
        losses = []
        B, N = targets.shape
        device = targets.device
        scale = self.train_loader.dataset.scale
        capacities = torch.tensor(get_capacities_from_graph(topology), device=device) / scale
        commodities_to_paths = mask_invalid_paths(self.commodities_to_paths, self.paths_to_edges, capacities)
        capacities = capacities.float().unsqueeze(-1)

        for i in range(B): 
            split_ratios = x[[i]] 
            current_tm = targets[[i]] 
            split_ratios = torch.transpose(split_ratios, 0, 1) 

            commodity_total_weight = commodities_to_paths.matmul(split_ratios)
            paths_over_total = commodities_to_paths.transpose(0, 1).matmul(1.0 / commodity_total_weight)
            split_ratios = split_ratios.mul(paths_over_total)

            tmp_demand_on_paths = commodities_to_paths.transpose(0, 1).matmul(current_tm.transpose(0, 1)) #shape: (num_paths, 1)
            demand_on_paths = tmp_demand_on_paths.mul(split_ratios) #shape: (num_paths, 1)
            flow_on_edges = self.paths_to_edges.transpose(0, 1).matmul(demand_on_paths) #shape: (num_edges, 1)
            congestion = flow_on_edges[capacities!=0].divide(capacities[capacities!=0]) #shape: (num_edges, 1)
            max_cong = torch.max(congestion.flatten(), dim = 0).values

            if train:
                loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong / max_cong.item()
                losses.append(loss)
            else:
                loss = max_cong.item()
                losses.append(loss)

        if train:
            return sum(losses) / len(losses) 
        else:
            return losses

    def load_checkpoint(self, path):
        """
        Load a model checkpoint.
        
        Args:
            path: Path to the checkpoint file
        """
        trainable_state_dict = torch.load(path + '/' + 'checkpoint.pt', map_location=self.accelerator.device)
        model_state_dict = self.model.state_dict()

        for name, param in trainable_state_dict.items(): 
            if name in model_state_dict: 
                model_state_dict[name].copy_(param)
            elif name.replace("module.", "") in model_state_dict:
                model_state_dict[name.replace("module.", "")].copy_(param)
            else: 
                self.accelerator.print(f'Not load {name}')

        self.model.load_state_dict(model_state_dict)
