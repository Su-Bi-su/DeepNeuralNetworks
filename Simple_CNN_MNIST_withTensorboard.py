# Imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.tensorboard import SummaryWriter
import torchvision

writer = SummaryWriter('runs/Simple_CNN_mnist_withTensorboard')


# Design model (Network)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_class=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyper-parameters

in_channels = 1
num_class = 10
learning_rate = 0.01
batch_size = 4
num_epoch = 1

# Load the data
train_dataset = Datasets.MNIST(root='dataset/', train=True, download=True, transform=transform.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Datasets.MNIST(root='dataset/', train=False, download=True, transform=transform.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# constant for classes
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Initialize the network
model = SimpleCNN().to(device)   # model in gpu
net = SimpleCNN()                # model in local device

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

# write network to to tensorboard
writer.add_graph(net, images)           # model is moved to 'cpu' to make sure that the model is in local device for tensorboard'
writer.close()


# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# Train Network
cnt = 0
running_loss = 0.0
for epoch in range(num_epoch):
    for batch_idx, data in enumerate(train_loader):
        cnt = cnt + 1
        images, labels = data
        # move data to cuda if available
        input_data = images.to(device)
        target = labels.to(device)

        # forward
        output = model(input_data)
        loss = criterion(output, target)
        if batch_idx % 50 == 0:
            cnt = cnt + 1
            writer.add_scalar("Loss/train", loss, cnt)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 1000 == 999:  # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(train_loader) + batch_idx)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, images, labels),
                              global_step=epoch * len(train_loader) + batch_idx)
            running_loss = 0.0


writer.close()


# check accuracy on training and test data

'''
def check_accuracy(loader, model):
    if loader == train_loader:
        print("Checking accuracy in train data.")
    else:
        print("Check accuracy on test data.")

    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            _, prediction = output.max(1)
            total += labels.size(0)
            correct += (prediction == labels).sum()
        print(f'Got {correct} / {total} with accuracy {float(correct) / float(total) * 100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

'''
