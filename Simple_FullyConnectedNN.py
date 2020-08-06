# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F               # all the functions has do not have any parameters like activation functions
import torch.optim as optim
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Design Fully Connected Network


class SimpleFullyConnectedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=250)
        self.fc3 = nn.Linear(in_features=250, out_features=50)
        self.fc4 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, input_data):
        fc1_output = F.relu(self.fc1(input_data))
        fc2_output = F.relu(self.fc2(fc1_output))
        fc3_output = F.relu(self.fc3(fc2_output))
        fc4_output = F.relu(self.fc4(fc3_output))
        return fc4_output


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Load Data

train_dataset = Datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = SimpleFullyConnectedNN(input_size=input_size, num_classes=num_classes).to(device=device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
train_loss = []
for each_epoch in range(num_epochs):
    print(f"Epoch : {each_epoch}")
    for batch_idx, data in enumerate(train_loader):

        input_data, labels = data

        # Move data to CUDA if possible
        data = input_data.to(device=device)
        targets = labels.to(device=device)

        # change the data to correct shape
        data = data.view(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        #print(f"Loss at batch idx : {batch_idx} is {loss.item() : .3f}")
        train_loss.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update parameters using adam
        optimizer.step()

# plot the training loss curve
fig, ax = plt.subplots()
ax.plot(range(len(train_loss)), train_loss)
ax.set_title("Loss Curve")
plt.xlabel("#epoches")
plt.ylabel("loss")
plt.show()


# Check accuracy on training and test data

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        in_data, labels = data
        in_data = in_data.to(device)
        in_data = in_data.view(in_data.shape[0], -1)
        labels = labels.to(device)

        outputs = model(in_data)
        _, prediction = outputs.max(1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

        print(f'Got {correct} / {total} with accuracy {float(correct)/float(total)*100:.2f}')
