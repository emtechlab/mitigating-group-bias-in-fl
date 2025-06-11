import copy
from .nodes.client import *
from .nodes.master import *
import random

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)
random.seed(seed)

def runner_train(args, client_loaders, test_loader, device, epoch, r_k_dic, lambda_reg, weight_coefficient_p):
    
    local_params = {}
    local_models = {}
    local_params_list = []
    local_grad_norms = []

    if (epoch == 1):

        global clients
        global master

        clients = {}

        for client_id in range(args.n_clients):

            clients[client_id] = define_localnode(args, device, client_id)

        master = define_globalnode(device, args)

    # Step 1: Calculate the 20% of the dictionary size
    sample_size = int(len(clients) * args.sampled_clients)
    
    # Step 2: Randomly sample the dictionary keys
    sampled_keys = random.sample(list(clients.keys()), sample_size)

    # Step 3: Create a new dictionary with keys starting from 0
    sampled_clients = {i: clients[key] for i, key in enumerate(sampled_keys)}

    for client_id, client in sampled_clients.items():

        client.client_id = client_id

        args.client_id = client_id

        # Distribute global weight to client
        global_weight = master.distribute_weight()
        copied_global_weight = copy.deepcopy(global_weight)

        local_param, local_model, avg_grad_norm = client.localround(client_loaders, test_loader, copied_global_weight, epoch, r_k_dic, lambda_reg, weight_coefficient_p)
        
        local_params[client_id]  = local_param
        local_models[client_id]  = local_model

        local_params_list.append(local_param)
        local_grad_norms.append(avg_grad_norm)

    master.aggregate(local_params, epoch)
    
    if (epoch == args.global_epochs):
        
        print("\nFinal Results")
        
        for client_id, client in sampled_clients.items():
            
            global_weight = master.distribute_weight()
            copied_global_weight = copy.deepcopy(global_weight)
            local_param = client.localround(copied_global_weight, epoch, validation_only=True)

    return