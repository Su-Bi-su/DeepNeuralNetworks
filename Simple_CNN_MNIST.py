# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transform


# Design model (Network)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_class=10 ):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self. conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


# Set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyper-parameters

in_channels = 1
num_class = 10
learning_rate = 0.01
batch_size = 64
num_epoch = 1

# Load the data
train_dataset = Datasets.MNIST(root='dataset/', train=True, download=True, transform=transform.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Datasets.MNIST(root='dataset/', train=False, download=True, transform=transform.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the network
model = SimpleCNN().to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epoch):
    for batch_idx, data in enumerate(train_loader):

        images, labels = data

        # move data to cuda if available
        input_data = images.to(device)
        target = labels.to(device)

        # forward
        output = model(input_data)
        loss = criterion(output, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()


# check accuracy on training and test data


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
        print(f'Got {correct} / {total} with accuracy {float(correct)/float(total)*100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

















