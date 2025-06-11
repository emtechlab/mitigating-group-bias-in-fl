import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pylab
import torch.nn as nn

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
        
def plot_confusion_matrix(client_id, cm, classes, epoch ,normalize=True, title='', cmap=plt.cm.Blues):

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        FONTSIZE = 25

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Specify the base name and the number to be appended
        base_name = '/home/khotso/FedGlobal/saved/matrix'
        
        # Generate the file name with the number appended
        file_name = f'{base_name}{client_id}{epoch}.npy'
        
        # Save the NumPy array to a file
        np.save(file_name, cm)

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

def local_validate_cifar10(args, model, device, epoch, testDataloader, criterion, client_id):
	
	model.eval()
	model.to(device)

	actual_labels = torch.tensor([])
	test_preds = torch.tensor([])

	correct = 0
	batch_loss = []

	with torch.no_grad():

		for batch_idx, (images, labels) in enumerate(testDataloader):
		
			images, labels = images.to(device), labels.to(device)
			
			actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)
			
			output = model(images)

			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(labels.view_as(pred)).sum().item()

			loss = criterion(output, labels)

			test_preds = torch.cat((test_preds.to(device), output) ,dim=0)

			batch_loss.append(loss.item())
	
	test_acc = 100. * correct / len(testDataloader.dataset)
	test_loss = sum(batch_loss)/len(batch_loss)

	preds_correct = test_preds.argmax(dim=1).eq(actual_labels).sum().item()

	if (args.test_adv == 0):
		
		cm = confusion_matrix(actual_labels.detach().cpu().numpy(), test_preds.argmax(dim=1).detach().cpu().numpy())

		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		plt.figure(figsize=(10,9))
		plot_confusion_matrix(client_id, cm, classes, epoch)

	print('| Client id:{} | Test_Loss: {:.3f} | Test_Acc: {:.3f} |'.format(client_id, test_loss, test_acc))

	return test_acc, test_loss
	
def local_validate_mnist(args, model, device, epoch, testDataloader, criterion, client_id):
	
	model.eval()
	model.to(device)

	actual_labels = torch.tensor([])
	test_preds = torch.tensor([])

	correct = 0
	batch_loss = []

	with torch.no_grad():

		for batch_idx, (images, labels) in enumerate(testDataloader):

			images, labels = images.to(device), labels.to(device)

			actual_labels = torch.cat((actual_labels.to(device), labels.type(torch.float).to(device)), dim=0)

			output = model(images)

			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(labels.view_as(pred)).sum().item()

			loss = criterion(output, labels)

			test_preds = torch.cat((test_preds.to(device), output) ,dim=0)

			batch_loss.append(loss.item())

	test_acc = 100. * correct / len(testDataloader.dataset)
	test_loss = sum(batch_loss)/len(batch_loss)

	preds_correct = test_preds.argmax(dim=1).eq(actual_labels).sum().item()

	if (args.test_adv == 0):
		
		cm = confusion_matrix(actual_labels.detach().cpu().numpy(), test_preds.argmax(dim=1).detach().cpu().numpy())

		classes = ('Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')

		plt.figure(figsize=(10,9))
		plot_confusion_matrix(client_id, cm, classes, epoch)

	print('| Client id:{} | Test_Loss: {:.3f} | Test_Acc: {:.3f} |'.format(client_id, test_loss, test_acc))

	return test_acc, test_loss
