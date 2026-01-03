import torch


def prepare_prompts(
    topology, 
    histories, 
    num_paths=8,
    bursty=None, 
    additional_info=None 
):
    """
    Prepare text prompts for the LLM model based on network topology and traffic histories.
    
    Args:
        topology: Network topology with node and edge information
        histories (Tensor): Traffic matrix histories of shape [batch_size, seq_length, num_od_pairs]
        num_paths (int): Number of paths per OD pair to consider as tunnels
        bursty (Tensor): Information about possible traffic bursts (optional)
        additional_info (str): Additional information to include in the prompt (optional)
        
    Returns:
        List[str]: A list of text prompts, one for each sample in the batch
    """
    # Get batch size from histories tensor
    B = histories.shape[0] 
    
    # Extract network information
    failed_links = get_failed_link_ids(topology)  # Get IDs of links with zero capacity
    num_nodes, num_links = len(topology.nodes()), len(topology.edges())
    
    # Compute overall trend of traffic matrices
    overall_trend = compute_overall_trend(histories)

    # Compute statistics for traffic matrices
    min_values = torch.amin(histories, dim=[1, 2])  # Minimum across sequence and OD pairs
    max_values = torch.amax(histories, dim=[1, 2])  # Maximum across sequence and OD pairs
    mean_values = torch.mean(histories, dim=[1, 2])  # Mean across sequence and OD pairs

    # Set default bursty values if not provided
    bursty = bursty if bursty is not None else torch.zeros_like(mean_values)

    # Format link failures information for the prompt
    if len(failed_links) == 0: 
        link_failures_prompt_ = "There are no link failures in the network" 
    else: 
        link_failures_prompt_ = f"Link Failures: {', '.join(map(str, failed_links))}"

    # Format additional information for the prompt
    if additional_info is None:
        conclusion_prompt_ = "When such events occur, the routing strategy must adjust accordingly"
    else:
        conclusion_prompt_ = f"{additional_info}When such events occur, the routing strategy must adjust accordingly"

    # Generate prompts for each sample in the batch
    prompts = [] 
    for b in range(B): 
        # Determine if there are traffic bursts for this sample
        if bursty[b] != 0: 
            traffic_bursts_prompt_ = "There are possible bursts in the future" 
        else:
            traffic_bursts_prompt_ = "The burst is unlikely to occur in the future" 

        # Construct the full prompt with all relevant information
        prompt = (
            f"<|start_prompt|>Goal: You are a WAN traffic engineer. Your task is to generate a routing strategy using the provided information. \
              The strategy should minimize the MLU. " 
            f"Description: there are {str(num_nodes)} nodes and {str(num_links)} edges in the WAN; "
            f"the shortest {num_paths} paths are selected as tunnels; " 
            f"the overall trend of historical TMs is {'upward' if overall_trend[b] > 0 else 'downward'}; " 
            f"the value of minimum, maximum, mean is {min_values[b]}, {max_values[b]}, and {mean_values[b]} respectively; " 
            "Domain Knowledge: " 
            f"{link_failures_prompt_}; " 
            f"{traffic_bursts_prompt_}; " 
            f"{conclusion_prompt_}<|end_prompt|>"
        ) 

        prompts.append(prompt)

    return prompts


def get_failed_link_ids(topology): 
    """
    Identify links in the topology that have failed (capacity = 0).
    
    Args:
        topology: Network topology with node and edge information
        
    Returns:
        List of tuples representing failed links as (source_node, target_node)
    """
    links = [] 
    for u, v, data in topology.edges(data=True): 
        if float(data['capacity']) == 0.: 
            links.append((u, v)) 
    return links


def compute_overall_trend(histories):
    """
    Compute the overall trend for each sample in the input tensor.

    Args:
        histories (Tensor): Input tensor of shape [num_samples, seq_length, num_pairs],
                            representing the historical values of OD pairs over time.

    Returns:
        overall_trend (Tensor): A tensor of shape [num_samples,], where each element is:
                                - 1 if the majority of OD pairs have a positive trend,
                                - -1 otherwise.
    """
    # Compute the trend for each OD pair: take temporal difference and sum over time
    trends = histories.diff(dim=1).sum(dim=1)  # Shape: [num_samples, num_pairs]

    # Count how many OD pairs have positive trends for each sample
    num_positive = (trends > 0).sum(dim=1)
    num_pairs = trends.size(1)

    # If more than half of the OD pairs are positive, set overall trend to 1; else -1
    overall_trend = torch.where(
        num_positive > (num_pairs / 2),
        torch.ones_like(num_positive, dtype=torch.float),
        -torch.ones_like(num_positive, dtype=torch.float)
    )

    return overall_trend