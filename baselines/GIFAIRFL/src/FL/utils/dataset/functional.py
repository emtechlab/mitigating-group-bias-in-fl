import numpy as np
import pandas as pd
import warnings

def split_indices(num_cumsum, rand_perm):
    """Splice the sample index list given number of each client.

    Args:
        num_cumsum (np.ndarray): Cumulative sum of sample number for each client.
        rand_perm (list): List of random sample index.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    client_indices_pairs = [(cid, idxs) for cid, idxs in enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    
    return client_dict

def balance_split(num_clients, nun_samples):

	""" Assign same sample for each client.

	Args:
	   num_clients (int): number of clients for partition.
	   num_samples (int): Total number of samples

	Returns:
	   numpy.ndarray: A numpy array consisting 'num_clients' integer elements, each represents sample number of corresponding clients.
	"""

	num_samples_per_client = int(nun_samples / num_clients)
	client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)

	return client_sample_nums

def homo_partition(client_sample_nums, num_samples):

	""" Partition data indices in IID way given sample numbers for each client

	Args:
	    client_sample_nums (numpy.ndarray): Sample numbers for each client
	    num_samples (int): Number of samples

	Returns:
	    dict: '{client_id: indices}'
	"""

	rand_perm = np.random.permutation(num_samples)
	num_cumsum = np.cumsum(client_sample_nums).astype(int)
	client_dict = split_indices(num_cumsum, rand_perm)

	return client_dict