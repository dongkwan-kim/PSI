from pathlib import Path
import sys

sys.path.insert(0, '..') # add config to path

# Output directory ('density' as an example)
DATASET_DIR = Path("/mnt/nas2/jiho/em_user_8")

# Flags
GENERATE_SYNTHETIC_G = False # whether to generate synthetic graph with below specified properties
GENERATE_NODE_EMB = True # whether to generate node embeddings

# Random Seed
RANDOM_SEED = 42

MAX_TRIES = 100

# Parameters for training node embeddings for base graph
CONV = "graphsaint_gcn"
MINIBATCH = "GraphSaint"
POSSIBLE_BATCH_SIZES = [512, 1024]
POSSIBLE_HIDDEN = [128, 256]
POSSIBLE_OUTPUT = [64]
POSSIBLE_LR = [0.001, 0.005]
POSSIBLE_WD = [5e-4, 5e-5]
POSSIBLE_DROPOUT = [0.4, 0.5]
POSSIBLE_NB_SIZE = [-1]
POSSIBLE_NUM_HOPS = [1]
POSSIBLE_WALK_LENGTH = [32]
POSSIBLE_NUM_STEPS = [32]
EPOCHS = 100

# Flags for precomputing similarity metrics
CALCULATE_SHORTEST_PATHS = True # Calculate pairwise shortest paths between all nodes in the graph
CALCULATE_DEGREE_SEQUENCE = True # Create a dictionary containing degrees of the nodes in the graph
CALCULATE_EGO_GRAPHS = True # Calculate the 1-hop ego graph associated with each node in the graph
OVERRIDE = False # Overwrite a similarity file even if it exists
N_PROCESSSES = 8 # Number of cores to use for multi-processsing when precomputing similarity metrics
