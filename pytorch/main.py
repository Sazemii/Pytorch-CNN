import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),  # converts PIL Images to PyTorchSensors from 0,255 to 0-1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # re-centers each RGB channel to have mean 0.5 and std 0.5, shifting values to
    # roughly [-1.0, 1.0] — this helps training converge faster
])

"""downloads and loads the CIFAR-10 dataset, which contains 60,000 color 
images (32×32 pixels) across 10 classes (airplane, car, bird, cat, etc.)"""


train_data = torchvision.datasets.CIFAR10(
    root='/data', train=True, transform=transform, download=True)
# 50,000 images for training

test_data = torchvision.datasets.CIFAR10(
    root='/data', train=False, transform=transform, download=True)
# 10,000 images for evaluation

# Dataloaders -> wraps the datasets to enable efficient batched iteration

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=2)
# feeds 32 images at a time to the model, randomizes order each epoch, 2 background processes to load data
# in parallel reducing load in gpu
# num_workers=2 means 2 separate CPU processes are preloading the next batch in the background
# while the GPU is still training on the current one — without this, the GPU would sit idle
# waiting for data after every batch, slowing down the whole training loop

"""Shuffle is important because -> Prevents the model from learning order patterns, Reduces gradient bias (for ex first batch is cats, the weights get pushed to that
, then upon encountering batch of dogs it overcorrects then forgets the cats, -> better balance), """

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=True, num_workers=2)

image, label = train_data[0]
print(image.size())  # prints torch.Size([3,32,32])


class_names = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Conv2d(in_channels, out_channels/filters, kernel_size)
        # Input image: (3, 32, 32) — 3 RGB channels, 32x32 pixels

        self.conv1 = nn.Conv2d(3, 12, 5)
        # Applies 12 filters of size 5x5 to the RGB image
        # Each filter scans the image and produces its own feature map
        # Spatial size: 32 - 5 + 1 = 28
        # Output shape: (12, 28, 28) — 12 feature maps, each 28x28

        self.pool = nn.MaxPool2d(2, 2)
        # Downsamples each feature map by taking the max in every 2x2 region
        # Reduces spatial size by half: 28 -> 14
        # Output shape after pool: (12, 14, 14)
        # Makes the model faster and more robust to small shifts in the image

        self.conv2 = nn.Conv2d(12, 24, 5)
        # Takes the 12 feature maps and applies 24 filters of size 5x5
        # More filters = learns more complex/abstract patterns than conv1
        # Spatial size: 14 - 5 + 1 = 10
        # Output shape: (24, 10, 10)
        # After pool again: (24, 5, 5)

        # --- Flatten: (24, 5, 5) -> 24 * 5 * 5 = 600 ---
        # Converts the 3D feature maps into a 1D vector for the fully connected layers

        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        # First fully connected layer
        # Takes the 600 flattened values, outputs 120 neurons
        # Starts combining all the spatial features learned by the conv layers

        self.fc2 = nn.Linear(120, 84)
        # Compresses 120 neurons down to 84
        # Continues learning higher-level combinations of features

        self.fc3 = nn.Linear(84, 10)
        # Output layer — 10 neurons, one per class (e.g. 10 categories in CIFAR-10)
        # The neuron with the highest value = the model's predicted class

    def forward(self, x):
        # Conv -> ReLU (activate) -> Pool (shrink): (3,32,32) -> (12,14,14)
        x = self.pool(F.relu(self.conv1(x)))
        # Conv -> ReLU (activate) -> Pool (shrink): (12,14,14) -> (24,5,5)
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten 3D -> 1D: (24,5,5) -> 600
        x = torch.flatten(x, 1)

        # Fully connected layers — learn from flattened features
        x = F.relu(self.fc1(x))  # 600 -> 120
        x = F.relu(self.fc2(x))  # 120 -> 84
        # 84 -> 10 (class scores, no activation needed)
        x = self.fc3(x)
        return x


net = NeuralNet()

loss_function = nn.CrossEntropyLoss()
# loss = (actual value - predicted value)^2
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# model updates weights in the direction that minimizes this loss
# momentum smoothens the oscillations
# Momentum adds a fraction of the previous update to the current one, simulating inertia.

"""Imagine a ball rolling down a hill towards the valley (global minimum).
 In plain SGD, the ball rolls but may get stuck in small bumps or oscillate back and forth, slowing down the descent. 
 In SGD with Momentum, however, the ball gains speed and is more likely to skip over small bumps, reaching the valley faster."""
if __name__ == '__main__':
    # The dataset and loader definitions above are at module level — they run even when this file
    # is just imported by another script. Re-defining them here inside __main__ ensures they only
    # actually execute when you intentionally run this file directly, preventing accidental
    # downloads or heavy memory use on import.
    train_data = torchvision.datasets.CIFAR10(
        root='/data', train=True, transform=transform, download=True)

    test_data = torchvision.datasets.CIFAR10(
        root='/data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=True, num_workers=2)

    net = NeuralNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(30):
        print(f'Training epoch {epoch}...')
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Loss: {running_loss / len(train_loader):.4f}')

    torch.save(net.state_dict(), 'trained_net.pth')
    # state_dict is a Python dictionary that maps every layer name to its current tensor of weights and biases
    # saving only the state_dict (not the whole model object) is the recommended practice because
    # conv1.weight -> [tensor of learned values]
    # conv1.bias   -> [tensor of learned values]
    # conv2.weight -> [tensor of learned values]
    # ...and so on

    net = NeuralNet().to(device)
    net.load_state_dict(torch.load('trained_net.pth', weights_only=True))
    # loads the saved weight snapshot back into a fresh NeuralNet instance
    # weights_only=True is a safety flag, it tells PyTorch to only restore tensors and skip
    # any arbitrary Python objects that could be embedded in the file (prevents code injection)

    correct = 0
    total = 0

    net.eval()
    # switches the model from training mode to evaluation mode
    # some layers behave differently during training vs inference, for example, Dropout randomly
    # zeros out neurons during training to prevent overfitting, but should be disabled at test time
    # so every neuron contributes to the prediction. BatchNorm similarly uses running statistics
    # instead of batch statistics. net.eval() flips both of these off so results are deterministic.

    with torch.no_grad():
        # by default PyTorch tracks every operation on tensors so it can compute gradients
        # during backpropagation — this tracking uses extra memory and compute
        # since we're just evaluating (not training), we don't need gradients at all
        # torch.no_grad() turns off that tracking entirely, making inference faster and lighter
        for data in test_loader:
            images, labels = data
            images, labels = images.to(
                device), labels.to(device)  # move to GPU
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            # torch.max(outputs, 1) scans across dimension 1 (the 10 class scores) and returns
            # two tensors: the highest score value, and the index where that score lives
            # the index IS the predicted class (e.g. index 3 = 'cat')
            # we don't care about the score itself, only which class won, so we discard the
            # first return value with _ (a Python convention for "I'm throwing this away")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

    new_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def load_image(image_path):
        image = Image.open(image_path)
        image = new_transform(image)
        image = image.unsqueeze(0)
        return image

    image_paths = ['example1.jpg', 'example2.jpg']
    images = [load_image(img) for img in image_paths]

    net.eval()
    with torch.no_grad():
        for image in images:
            image = image.to(device)
            output = net(image)
            _, predicted = torch.max(output, 1)
            print(f'Prediction: {class_names[predicted.item()]}')
