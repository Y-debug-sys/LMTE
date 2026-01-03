# Configuration variables for the LMTE (Large Language Model for Traffic Engineering) experiment on GEANT topology
GPUs=4
PORT=$((29500 + RANDOM % 1000))

# Model and training parameters
batch_size=32
eval_batch_size=32
num_tunnels=8                    # Number of paths per OD pair
llama_layers=8                   # Number of LLaMA model layers to use
history_length=12                # Length of historical traffic matrix sequence
learning_rate=0.001
dropout=0.25
normlized_scale=1000000000       # Normalization scale factor

# First experiment: Train the model on GEANT topology
# Uses accelerate for multi-GPU training with mixed precision
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $GPUs --main_process_port $PORT main.py \
  --topology GEANT\                                    # Network topology to use
  --topology_filepath ./data/GEANT/topology.json\      # Path to topology file
  --tm_filepath ./data/GEANT/GEANT.csv\                # Path to traffic matrix file
  --is_training 1\                                     # Enable training mode
  --num_itrs 3\                                        # Number of experiment iterations
  --train_epochs 10\                                   # Number of training epochs
  --patience 3\                                        # Early stopping patience
  --d_keys 32\                                         # Dimension of keys in attention
  --d_model 32\                                        # Model dimension
  --llm_model llama-8b\                                # LLM model type
  --llm_dim '4096'\                                    # LLM model dimension
  --batch_size $batch_size\                            # Training batch size
  --eval_batch_size $eval_batch_size\                  # Evaluation batch size
  --llm_layers $llama_layers\                          # Number of LLM layers
  --scale $normlized_scale\                            # Normalization scale
  --num_paths $num_tunnels\                            # Number of paths per OD pair
  --window_size $history_length\                       # Window size for history
  --learning_rate $learning_rate\                      # Learning rate

# Second experiment: Test the trained model with link failures
# Uses the same parameters but in testing mode with failure injection
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $GPUs --main_process_port $PORT main.py \
  --topology GEANT\                                    # Network topology to use
  --topology_filepath ./data/GEANT/topology.json\      # Path to topology file
  --tm_filepath ./data/GEANT/GEANT.csv\                # Path to traffic matrix file
  --is_training 0\                                     # Disable training mode (testing only)
  --num_itrs 3\                                        # Number of experiment iterations
  --train_epochs 10\                                   # Number of training epochs (not used in testing)
  --patience 3\                                        # Early stopping patience (not used in testing)
  --d_keys 32\                                         # Dimension of keys in attention
  --d_model 32\                                        # Model dimension
  --llm_model llama-8b\                                # LLM model type
  --llm_dim '4096'\                                    # LLM model dimension
  --batch_size $batch_size\                            # Batch size
  --add_failures 1\                                    # Enable link failure injection
  --num_failures 1\                                    # Number of link failures to inject
  --eval_batch_size $eval_batch_size\                  # Evaluation batch size
  --llm_layers $llama_layers\                          # Number of LLM layers
  --scale $normlized_scale\                            # Normalization scale
  --num_paths $num_tunnels\                            # Number of paths per OD pair
  --window_size $history_length\                       # Window size for history
  --learning_rate $learning_rate\                      # Learning rate (not used in testing)

# Third experiment: Test the trained model with traffic bursts
# Uses the same parameters but in testing mode with burst injection
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $GPUs --main_process_port $PORT main.py \
  --topology GEANT\                                    # Network topology to use
  --topology_filepath ./data/GEANT/topology.json\      # Path to topology file
  --tm_filepath ./data/GEANT/GEANT.csv\                # Path to traffic matrix file
  --is_training 0\                                     # Disable training mode (testing only)
  --num_itrs 3\                                        # Number of experiment iterations
  --train_epochs 10\                                   # Number of training epochs (not used in testing)
  --patience 3\                                        # Early stopping patience (not used in testing)
  --d_keys 32\                                         # Dimension of keys in attention
  --d_model 32\                                        # Model dimension
  --llm_model llama-8b\                                # LLM model type
  --llm_dim '4096'\                                    # LLM model dimension
  --batch_size $batch_size\                            # Batch size
  --add_bursts 1\                                      # Enable traffic burst injection
  --burst_factor 10\                                   # Factor for traffic burst generation
  --eval_batch_size $eval_batch_size\                  # Evaluation batch size
  --llm_layers $llama_layers\                          # Number of LLM layers
  --scale $normlized_scale\                            # Normalization scale
  --num_paths $num_tunnels\                            # Number of paths per OD pair
  --window_size $history_length\                       # Window size for history
  --learning_rate $learning_rate\                      # Learning rate (not used in testing)