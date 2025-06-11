import argparse

def get_args():

    parser = argparse.ArgumentParser(description="Parameters for running training")

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['utk', 'cifar10', 'cifar100', 'mnist', 'fmnist', 'fer', 'acsemployment', 'acsincome'])
    parser.add_argument('--local_epochs', type=int, default=1, help='the number of local epochs') # 5
    parser.add_argument('--global_epochs', type=int, default=200, help='the number of federated learning rounds') # 30
    parser.add_argument('--n_clients', type=int, default=5, help='number of clients')

    parser.add_argument("--partition", type=str, required=True, choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=None, help="Dirichlet alpha (required for dirichlet)")
    parser.add_argument('--client_id', type=int, default=0, help='client id')

    parser.add_argument('--federated_type', type=str, default='fedavg', choices=['ditto', 'fedavg', 'fedasam', 'afl', 'fedadam', 'fedyogi', 'fedadagrad', 'fedprox'])
    parser.add_argument("--optimizer" , type=str, default="sgd", choices=['sgd','adam'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate') 
    parser.add_argument('--init_lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--rho', type=float, default=0.02, help='learning rate')
    parser.add_argument('--eta', type=float, default=0.02, help='learning rate')  
    parser.add_argument('--mu', type=float, default=0.1, help='learning rate') # mus=('0.01' '0.001')

    parser.add_argument('--dir_alpha', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dp_noise', type=float, default=0.01, help='differential privacy noise')
    parser.add_argument('--sampled_clients', type=float, default=0.2, help='number of sampled clients per round')

    parser.add_argument('--beta1', type=float, default=0.1, help='beta')
    parser.add_argument('--beta2', type=float, default=0.1, help='beta')

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--target_norm', type=float, default=0.0, help='target gradient norm') 
    parser.add_argument('--reg_lambda', type=float, default=7.1, help='regularization norm')     
    parser.add_argument('--base_kt', type=float, default=0.01, help='base kuramoto strength')

    parser.add_argument('--batch_size', type=int, default=64)  # 256
    parser.add_argument('--data_seed', type=int, default=2024)  # 256 
    parser.add_argument('--clip_value', default=60.00, type=float)
    parser.add_argument('--reduction_factor', default=1.00, type=float)

    parser.add_argument('--t_value', type=int, default=-1)  
    parser.add_argument('--noise', default=0.00, type=float)

    parser.add_argument('--drfa_gamma', default=0.01, type=float)

    args = parser.parse_args()

    return args