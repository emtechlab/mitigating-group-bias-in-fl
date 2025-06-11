import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pylab
import torchattacks
from torch import nn

import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

# Define the architecture for noise estimation and denoising
class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        # Add layers for noise estimation and denoising (this is a simplified example)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.estimation_layer = nn.Linear(128 * 8 * 8, 256)
        self.denoising_layer = nn.Linear(256, 128 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        estimated_noise = self.estimation_layer(features)
        denoised = self.denoising_layer(estimated_noise)
        denoised = denoised.view(-1, 128, 8, 8)
        reconstructed = self.decoder(denoised)
        return reconstructed, estimated_noise
                
# Define a custom loss function that handles NaN values
class WeightedCrossEntropyLoss(nn.Module):
    
    def __init__(self, weight=None, reduction='mean'):
        
        super(WeightedCrossEntropyLoss, self).__init__()
        
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        
        log_softmax = nn.functional.log_softmax(input, dim=1)
        loss = nn.functional.nll_loss(log_softmax, target, weight=self.weight, reduction=self.reduction)
        
        # Handle NaN loss values
        if torch.isnan(loss).any():
            loss = torch.tensor(0.0, requires_grad=True)
        
        return loss

def plot_confusion_matrix(client_id, cm, classes, epoch ,normalize=True, title='', cmap=plt.cm.Blues):

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        FONTSIZE = 25

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=FONTSIZE-8)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=FONTSIZE-6)
        plt.xlabel('Predicted label', fontsize=FONTSIZE-6)

        plt.xticks(fontsize=FONTSIZE-6)
        plt.yticks(fontsize=FONTSIZE-6)

        cm_dest = "/home/khotso/FedGlobal/saved/confusion_matrix" + str(client_id) + str(epoch) + ".png"
        pylab.savefig(cm_dest, bbox_inches='tight', pad_inches=0)

def update_weights_cifar10(args, model, device, global_epoch, trainDataloader, testDataloader, criterion, client_id):
	
	if (args.optimizer == 'sgd'):

		if (args.federated_type == 'mwur') or (args.federated_type == 'fedind'):

			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
		else:

			#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
		
		#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

	elif (args.optimizer == 'adam'):

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

	for epoch in range(1, args.local_epochs+1):
		
		batch_loss = []
		correct = 0

		model.train()
		model.to(device)

		loss_all = 0
		total = 0
		correct = 0
		adv_acc_all = 0
		
		current_loss = 0
		
		actual_labels = torch.tensor([])
		test_preds = torch.tensor([])

		if (args.federated_type == 'mwur'):
		
			r_a_k_list = [0.0] * 10
			
			p_a = torch.load('/home/khotso/image_classification_mnist/saved/weighting/p_a.pt')
			mu = torch.load('/home/khotso/image_classification_mnist/saved/weighting/mu.pt')
			w_a = mu / p_a
			
			w_a_weight = w_a.tolist()
			
		for batch_idx, (images, labels) in enumerate(trainDataloader):

			class_count_list = [0] * 10  # There are 10 classes in CIFAR-10
			
			for label in labels:
				
				class_count_list[label] += 1
				
			if (args.federated_type == 'mwur'):
				
				# Normalize importance weights
				weight_sum = sum(w_a_weight)
				
				for k in range(len(w_a_weight)):
					
					w_a_weight[k] = (w_a_weight[k]**(1 / (class_count_list[k] + 1))) / weight_sum
				
				criterion = WeightedCrossEntropyLoss(weight=torch.tensor(w_a_weight).to(device))

			images, labels = images.to(device), labels.to(device)
			
			r_k = 0

			optimizer.zero_grad()

			output = model(images)
			
			if (args.n_clients == 1):

				# Include the Softmax layer here -----------------> compute average likelihood estimates
				softmax_model = nn.Softmax(dim=1)
				soft_output = softmax_model(output)
				avg_likelihood = torch.mean(soft_output, 0).cpu().detach().numpy()

			actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)

			test_preds = torch.cat((test_preds.to(device), output) ,dim=0)

			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(labels.view_as(pred)).sum().item()

			if (args.federated_type == 'fedavg') | (args.federated_type == 'afl') | (args.federated_type == 'fedind'):

				loss_classification = criterion(output, labels)
			else:
				
				r_k = criterion(output, labels)
				loss =  r_k
				
			if (args.federated_type == 'fedind') and (client_id in [0, 1, 2, 3, 4]):
				
				# Calculate errors for each group
				group_errors = torch.zeros(10) # Assuming 10 groups for CIFAR-10
				group_counts = torch.zeros(10)
				lambda_k = torch.full((10,), 0.003)
				
				hyper = torch.full((10,), 0.0001)
				
				# Compute errors for each group
				for k in range(10):  # 10 groups in CIFAR-10
					
					group_indices = torch.where(labels == k)[0]
					group_predictions = pred[group_indices]
					group_targets = labels[group_indices]
					
					group_errors[k] += (group_predictions != group_targets).sum().item()
					group_counts[k] += len(group_indices)
					
				# Compute error rates for each group and calculate adaptive alphas
				group_error_rates = group_errors / (group_counts + 1e-8)  # To avoid division by zero
				adaptive_alphas = group_error_rates.max() / (group_error_rates + 1e-8)
				
				# Compute e to the power of the negative of each element
				adaptive_alphas = torch.exp(-4.0*adaptive_alphas)
				
				# Calculate fairness term with adaptive alpha for each group
				fairness_term = 0
				
				for k in range(10):  # 10 groups in CIFAR-10
					
					group_indices = torch.where(labels == k)[0]
					
					if (len(group_indices) != 0):
						
						group_loss = torch.mean(nn.functional.cross_entropy(output[group_indices], labels[group_indices]))
					else:
						
						group_loss = torch.tensor(0.0, requires_grad=True).to(device)
					
					fairness_term += torch.abs(((torch.sigmoid(group_loss) - torch.sigmoid(adaptive_alphas[k].to(device))   ) ** 2))
					#fairness_term += torch.abs((group_loss - hyper[k].to(device)   ) ** 2)
								
				# Total loss with fairness term
				#loss = (1 - lambda_k[0]) * loss_classification + lambda_k[0] * fairness_term
				loss = loss_classification + fairness_term
			
			else:
				
				loss = loss_classification
				
			loss.backward()
			optimizer.step()
        			
			if (args.federated_type == 'mwur'):
				
				# Calculate per-class losses
				for j in range(10):
					
					mask = (labels == j)
					r_a_k = torch.sum(mask.float() * loss) / torch.sum(mask.float() + 1e-10)
					r_a_k_list[j] += r_a_k.item()
		
			batch_loss.append(loss.item())

		train_acc,train_loss=100. * correct / len(trainDataloader.dataset),sum(batch_loss) / len(batch_loss)

		if (args.federated_type == 'mwur'):
			
			# Normalize per-class losses and store them
			r_a_ks = torch.tensor([class_loss / 12000 for class_loss in r_a_k_list])

		if (args.n_clients == 1):
			print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} | Likelihoods: |'.format(	
				global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc), avg_likelihood)
		else:
			print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} |'.format(
				global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc))

	if (args.federated_type == 'mwur'):

		return model.state_dict(), r_a_ks
	else:
		return model.state_dict()
		
def update_weights_mnist(args, model, device, global_epoch, trainDataloader, testDataloader, criterion, client_id):
	
	if (args.optimizer == 'sgd'):

		if (args.federated_type == 'mwur'):

			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
		else:

			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

	elif (args.optimizer == 'adam'):

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

	for epoch in range(1, args.local_epochs+1):
		
		batch_loss = []
		correct = 0

		model.train()
		model.to(device)

		loss_all = 0
		total = 0
		correct = 0
		adv_acc_all = 0
		
		current_loss = 0
		
		actual_labels = torch.tensor([])
		test_preds = torch.tensor([])

		if (args.federated_type == 'mwur'):
		
			r_a_k_list = [0.0] * 10
			
			p_a = torch.load('/home/khotso/image_classification_mnist/saved/weighting/p_a.pt')
			mu = torch.load('/home/khotso/image_classification_mnist/saved/weighting/mu.pt')
			w_a = mu / p_a
			
			w_a_weight = w_a.tolist()
			
		for batch_idx, (images, labels) in enumerate(trainDataloader):

			class_count_list = [0] * 10  # There are 10 classes in CIFAR-10
			
			for label in labels:
				
				class_count_list[label] += 1
				
			if (args.federated_type == 'mwur'):
				
				# Normalize importance weights
				weight_sum = sum(w_a_weight)
				
				for k in range(len(w_a_weight)):
					
					w_a_weight[k] = (w_a_weight[k]**(1 / (class_count_list[k] + 1))) / weight_sum
				
				criterion = WeightedCrossEntropyLoss(weight=torch.tensor(w_a_weight).to(device))

			images, labels = images.to(device), labels.to(device)

			r_k = 0

			optimizer.zero_grad()

			output = model(images)
			
			if (args.n_clients == 1):

				# Include the Softmax layer here -----------------> compute average likelihood estimates
				softmax_model = nn.Softmax(dim=1)
				soft_output = softmax_model(output)
				avg_likelihood = torch.mean(soft_output, 0).cpu().detach().numpy()

			actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)

			test_preds = torch.cat((test_preds.to(device), output) ,dim=0)

			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(labels.view_as(pred)).sum().item()

			if (args.federated_type == 'fedavg') | (args.federated_type == 'afl') | (args.federated_type == 'fedind'):

				loss_classification = criterion(output, labels)
			else:
				
				r_k = criterion(output, labels)
				loss =  r_k
				
			if (args.federated_type == 'fedind'):
				
				# Calculate errors for each group
				group_errors = torch.zeros(10) # Assuming 10 groups for CIFAR-10
				group_counts = torch.zeros(10)
				lambda_k = torch.full((10,), 0.003)
				
				# Compute errors for each group
				for k in range(10):  # 10 groups in CIFAR-10
					
					group_indices = torch.where(labels == k)[0]
					group_predictions = pred[group_indices]
					group_targets = labels[group_indices]
					
					group_errors[k] += (group_predictions != group_targets).sum().item()
					group_counts[k] += len(group_indices)
					
				# Compute error rates for each group and calculate adaptive alphas
				group_error_rates = group_errors / (group_counts + 1e-8)  # To avoid division by zero
				adaptive_alphas = group_error_rates.max() / (group_error_rates + 1e-8)
				
				# Compute e to the power of the negative of each element
				adaptive_alphas = torch.exp(-4.0*adaptive_alphas)
				
				# Calculate fairness term with adaptive alpha for each group
				fairness_term = 0
				
				for k in range(10):  # 10 groups in CIFAR-10
					
					group_indices = torch.where(labels == k)[0]
					
					if (len(group_indices) != 0):
						
						group_loss = torch.mean(nn.functional.cross_entropy(output[group_indices], labels[group_indices]))
					else:
						
						group_loss = torch.tensor(0.0, requires_grad=True).to(device)
					
					fairness_term += torch.abs(((torch.sigmoid(group_loss) - torch.sigmoid(adaptive_alphas[k].to(device))   ) ** 2))
								
				# Total loss with fairness term
				#loss = (1 - lambda_k[0]) * loss_classification + lambda_k[0] * fairness_term
				loss = loss_classification + fairness_term
			
			else:
				
				loss = loss_classification
				
			loss.backward()
			optimizer.step()
        			
			if (args.federated_type == 'mwur'):
				
				# Calculate per-class losses
				for j in range(10):
					
					mask = (labels == j)
					r_a_k = torch.sum(mask.float() * loss) / torch.sum(mask.float() + 1e-10)
					r_a_k_list[j] += r_a_k.item()
		
			batch_loss.append(loss.item())

		train_acc,train_loss=100. * correct / len(trainDataloader.dataset),sum(batch_loss) / len(batch_loss)

		if (args.federated_type == 'mwur'):
			
			# Normalize per-class losses and store them
			r_a_ks = torch.tensor([class_loss / 12000 for class_loss in r_a_k_list])

		if (args.n_clients == 1):
			print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} | Likelihoods: |'.format(	
				global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc), avg_likelihood)
		else:
			print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} |'.format(
				global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc))

	if (args.federated_type == 'mwur'):

		return model.state_dict(), r_a_ks
	else:
		return model.state_dict()
			
'''
def update_weights_cifar10(args, model, device, global_epoch, trainDataloader, testDataloader, criterion, client_id):

    if (args.optimizer == 'sgd'):

        if (args.federated_type == 'fedind'):

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #

        else:

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    elif (args.optimizer == 'adam'):

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Define fairness parameters
    #lambda_k = torch.ones(10)  # Lagrange multipliers for ten groups

    # Create a tensor with all values equal to 0.997 and size 10
    lambda_k = torch.full((10,), 1.00)
    lambda_1_k = torch.full((10,), 0.007)
    
    # Started with 0.001 --> discrepancy=26
    alpha = 0.0006  # Desired fairness level

    for epoch in range(1, args.local_epochs+1):

        batch_loss = []
        correct = 0

        model.train()
        model.to(device)

        loss_all = 0
        total = 0
        correct = 0
        adv_acc_all = 0
        clip_value = 1.0

        actual_labels = torch.tensor([])
        test_preds = torch.tensor([])

        for batch_idx, (images, labels) in enumerate(trainDataloader):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)

            loss_classification = criterion(output, labels)

            if (args.federated_type == 'fedind'):

                # Calculate fairness term
                fairness_term = 0

                for k in range(10):  # 10 groups in CIFAR-10

                    group_indices = torch.where(labels == k)[0]
                    group_loss = torch.mean(F.cross_entropy(output[group_indices], labels[group_indices])) 
                    #fairness_term += lambda_k[k] * torch.abs(((group_loss - alpha) ** 1)) 
                    fairness_term += lambda_k[k] * torch.abs(((group_loss - alpha) ** 1)) 

                # Total loss with fairness term
                loss = loss_classification + fairness_term

            else:

                loss = loss_classification

            actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)

            test_preds = torch.cat((test_preds.to(device), output) ,dim=0)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            # L1 regularization

            l1_reg = torch.tensor(0., requires_grad=True)

            for param in model.parameters():

                l1_reg = l1_reg + torch.norm(param, 1)  # L1 norm of the parameters

            loss = loss + args.regularization_param * l1_reg  # Add L1 regularization term to the loss

            loss.backward()

            for param in model.parameters():

                if param.grad is not None:

                    param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)

            optimizer.step()

            batch_loss.append(loss.item())

        train_acc,train_loss=100. * correct / len(trainDataloader.dataset),sum(batch_loss)/len(batch_loss)

        print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} |'.format(
                global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc))

    return model.state_dict()
'''

'''
def update_weights_mnist(args, model, device, global_epoch, trainDataloader, testDataloader, criterion, client_id):
	
	if (args.optimizer == 'sgd'):

		if (args.federated_type == 'mwur'):

			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
		else:

			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

	elif (args.optimizer == 'adam'):

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

	for epoch in range(1, args.local_epochs+1):
		
		batch_loss = []
		correct = 0

		model.train()
		model.to(device)

		loss_all = 0
		total = 0
		correct = 0
		adv_acc_all = 0

		actual_labels = torch.tensor([])
		test_preds = torch.tensor([])

		if (args.federated_type == 'mwur'):
		
			r_a_k_list = [0.0] * 10
			
			p_a = torch.load('/home/khotso/image_classification_mnist/saved/weighting/p_a.pt')
			mu = torch.load('/home/khotso/image_classification_mnist/saved/weighting/mu.pt')
			w_a = mu / p_a
			
			w_a_weight = w_a.tolist()
			
		for batch_idx, (images, labels) in enumerate(trainDataloader):

			class_count_list = [0] * 10  # There are 10 classes in CIFAR-10
			
			for label in labels:
				
				class_count_list[label] += 1
				
			if (args.federated_type == 'mwur'):
				
				# Normalize importance weights
				weight_sum = sum(w_a_weight)
				
				for k in range(len(w_a_weight)):
					
					w_a_weight[k] = (w_a_weight[k]**(1 / (class_count_list[k] + 1))) / weight_sum
				
				criterion = WeightedCrossEntropyLoss(weight=torch.tensor(w_a_weight).to(device))

			images, labels = images.to(device), labels.to(device)

			r_k = 0

			optimizer.zero_grad()

			output = model(images)
			
			if (args.n_clients == 1):

				# Include the Softmax layer here -----------------> compute average likelihood estimates
				softmax_model = nn.Softmax(dim=1)
				soft_output = softmax_model(output)
				avg_likelihood = torch.mean(soft_output, 0).cpu().detach().numpy()

			actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)

			test_preds = torch.cat((test_preds.to(device), output) ,dim=0)

			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(labels.view_as(pred)).sum().item()

			if (args.federated_type == 'fedavg') | (args.federated_type == 'afl') | (args.federated_type == 'fedind'):

				loss_classification = criterion(output, labels)
			else:
				
				r_k = criterion(output, labels)
				loss =  r_k
				
			if (args.federated_type == 'fedind'):
				
				# Calculate errors for each group
				group_errors = torch.zeros(10) # Assuming 10 groups for CIFAR-10
				group_counts = torch.zeros(10)
				lambda_k = torch.full((10,), 0.003)
				
				# Compute errors for each group
				for k in range(10):  # 10 groups in CIFAR-10
					
					group_indices = torch.where(labels == k)[0]
					group_predictions = pred[group_indices]
					group_targets = labels[group_indices]
					
					group_errors[k] += (group_predictions != group_targets).sum().item()
					group_counts[k] += len(group_indices)
					
				# Compute error rates for each group and calculate adaptive alphas
				group_error_rates = group_errors / (group_counts + 1e-8)  # To avoid division by zero
				adaptive_alphas = group_error_rates.max() / (group_error_rates + 1e-8)
				
				# Compute e to the power of the negative of each element
				adaptive_alphas = torch.exp(-3.0*adaptive_alphas)
				
				# Calculate fairness term with adaptive alpha for each group
				fairness_term = 0
				
				for k in range(10):  # 10 groups in CIFAR-10
					
					group_indices = torch.where(labels == k)[0]
					group_loss = torch.mean(nn.functional.cross_entropy(output[group_indices], labels[group_indices]))
					fairness_term += (group_loss - adaptive_alphas[k]) ** 2
					
				# Total loss with fairness term
				loss = (1 - lambda_k[0]) * loss_classification + lambda_k[0] * fairness_term
			
			else:
				
				loss = loss_classification
				
			loss.backward()
			optimizer.step()
        			
			if (args.federated_type == 'mwur'):
				
				# Calculate per-class losses
				for j in range(10):
					
					mask = (labels == j)
					r_a_k = torch.sum(mask.float() * loss) / torch.sum(mask.float() + 1e-10)
					r_a_k_list[j] += r_a_k.item()
		
			batch_loss.append(loss.item())

		train_acc,train_loss=100. * correct / len(trainDataloader.dataset),sum(batch_loss) / len(batch_loss)

		if (args.federated_type == 'mwur'):
			
			# Normalize per-class losses and store them
			r_a_ks = torch.tensor([class_loss / 12000 for class_loss in r_a_k_list])

		if (args.n_clients == 1):
			print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} | Likelihoods: |'.format(	
				global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc), avg_likelihood)
		else:
			print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} |'.format(
				global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc))

	if (args.federated_type == 'mwur'):

		return model.state_dict(), r_a_ks
	else:
		return model.state_dict()
'''










































'''
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pylab
import torchattacks
from torch import nn

from sklearn.metrics import confusion_matrix
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

class Minimax(object):

    def __init__(self):

        self.weight = 0.5

    def slice_outputs(self, label, scores, labels):

        indeces = torch.where(labels == label)
        indeces = indeces[0].detach().cpu()

        if (len(indeces) != 0):

            outputs = scores[indeces]
            targets = labels[indeces]
        else:

            outputs = 'empty'
            targets = 'empty'

        return outputs, targets

def plot_confusion_matrix(client_id, cm, classes, epoch ,normalize=True, title='', cmap=plt.cm.Blues):

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        FONTSIZE = 25

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=FONTSIZE-8)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=FONTSIZE-6)
        plt.xlabel('Predicted label', fontsize=FONTSIZE-6)

        plt.xticks(fontsize=FONTSIZE-6)
        plt.yticks(fontsize=FONTSIZE-6)

        cm_dest = "/home/khotso/FedGlobal/saved/confusion_matrix" + str(client_id) + str(epoch) + ".png"
        pylab.savefig(cm_dest, bbox_inches='tight', pad_inches=0)

def adversarial_attack(model, optimizer, num_classes, x_test, y_test, loss_func, device):

        # Create the ART classifier
        classifier = PyTorchClassifier(model=model, clip_values=(0.0, 1.0), loss=loss_func, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=num_classes)

        # Generate adversarial test examples
        attack = FastGradientMethod(estimator=classifier, eps=0.3)

        x_test_adv = attack.generate(x=x_test)

        # Evaluate the ART classifier on adversarial test examples
        predictions = classifier.predict(x_test_adv)

        out = torch.tensor(predictions).to(device)

        acc = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)

        return acc, out, x_test_adv

def compute_l2_loss(self, w):

    return torch.square(w).sum()

def update_weights_cifar10(args, model, device, global_epoch, trainDataloader, testDataloader, criterion, client_id):

    if (args.optimizer == 'sgd'):

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    elif (args.optimizer == 'adam'):

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.local_epochs+1):

        batch_loss = []
        correct = 0

        model.train()
        model.to(device)

        loss_all = 0
        total = 0
        correct = 0
        adv_acc_all = 0

        actual_labels = torch.tensor([])
        test_preds = torch.tensor([])

        if (args.federated_type == 'mwur'):

            r_a_k_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            p_a = torch.load('/home/khotso/FedGlobal/saved/weighting/p_a.pt')
            mu = torch.load('/home/khotso/FedGlobal/saved/weighting/mu.pt')
            minimax = Minimax()
            w_a = mu / p_a

        for batch_idx, (images, labels) in enumerate(trainDataloader):

            images, labels = images.to(device), labels.to(device)

            r_k = 0

            optimizer.zero_grad()

            output = model(images)

            if (args.n_clients == 1):

                # Include the Softmax layer here -----------------> compute average likelihood estimates
                softmax_model = nn.Softmax(dim=1)
                soft_output = softmax_model(output)
                avg_likelihood = torch.mean(soft_output, 0).cpu().detach().numpy()

            actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)

            test_preds = torch.cat((test_preds.to(device), output) ,dim=0)

            if (args.test_adv == 1):


                actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)

                #batch_acc, out, x_test_adv = adversarial_attack(model, optimizer, 10, images.cpu().numpy(), labels.cpu().numpy(), criterion, device)

                #adv_acc_all += batch_acc

                test_preds = torch.cat((test_preds.to(device), out) ,dim=0)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            if (args.federated_type == 'fedavg') | (args.federated_type == 'afl'):

                loss = criterion(output, labels)

            else:

                for i in range(10):

                    outputss, targetss = minimax.slice_outputs(i, output, labels)

                    if (outputss != 'empty'):

                        r_a_k = criterion(outputss, targetss)
                        r_a_k_list[i] = r_a_k_list[i] + r_a_k
                    else:

                        r_a_k = 0.0

                    r_k  = r_k + (r_a_k * w_a[i].item())

                r_k = r_k / 10

                loss =  r_k
                l1_lambda = args.regularization_param #0.00001
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            loss.backward()

            optimizer.step()

            batch_loss.append(loss.item())

        train_acc,train_loss=100. * correct / len(trainDataloader.dataset),sum(batch_loss)/len(batch_loss)

        if (args.federated_type == 'mwur'):

            r_a_ks = torch.tensor(r_a_k_list) / (batch_idx + 1)

        if (args.test_adv == 1):

            cm = confusion_matrix(actual_labels.detach().cpu().numpy(), test_preds.argmax(dim=1).detach().cpu().numpy())
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            plt.figure(figsize=(10,9))
            plot_confusion_matrix(client_id, cm, classes,global_epoch)

        if (args.n_clients == 1):

            print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} | Likelihoods: |'.format(
                global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc), avg_likelihood)

        else:
            print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f} |'.format(
                global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss, train_acc))

    if (args.federated_type == 'mwur'):

        return model.state_dict(), r_a_ks

    else:
        return model.state_dict()
'''
