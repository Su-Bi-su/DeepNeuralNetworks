# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


# Design Network
class ImageClassifierOnCIFAR(nn.Module):
    def __init__(self, in_channel=3, num_class=10):
        super(ImageClassifierOnCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Set the device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyper-parameters
in_channels = 3
no_class = 10
learning_rate = 0.001
batch_size = 4
epoch = 3

# Load Data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Dataset.CIFAR10(root='dataset/', train=True, download=True, transform=transform)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

testset = Dataset.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


# Initialize the network

model = ImageClassifierOnCIFAR(in_channel=3, num_class=no_class).to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the network
for each_epoch in range(epoch):
    for batch_idx, data in enumerate(trainloader):
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        # zero the parameters gradients
        optimizer.zero_grad()

        # forward
        output = model(images)
        loss = criterion(output, labels)
        if batch_idx % 10 == 0:
            writer.add_scalar("Train Loss", loss, batch_idx)

        # backward
        loss.backward()

        # update
        optimizer.step()

print("Finished Training!!")


# Compute accuracy of the network
def check_accuracy(loader, model):
    if loader == trainloader:
        print("Checking accuracy in train data.")
    else:
        print("Check accuracy on test data.")

    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for data in loader:
            img, target = data

            img = img.to(device)
            target = target.to(device)

            scores = model(img)
            _, prediction = scores.max(1)
            total += target.size(0)
            correct += (prediction == target).sum()

        print(f"{correct} / {total} were correct, which means the accuracy is : {float(correct) / float(total) * 100 :.2f}")


check_accuracy(trainloader, model)
check_accuracy(testloader, model)
