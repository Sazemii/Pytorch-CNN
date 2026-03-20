import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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


if __name__ == '__main__':
    # Just load the trained model — no training
    net = NeuralNet().to(device)
    net.load_state_dict(torch.load('trained_net.pth', weights_only=True))
    net.eval()

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

    image_paths = ['example1.jpg', 'example2.jpg', 'example3.jpg']

    with torch.no_grad():
        for path in image_paths:
            image = load_image(path).to(device)
            output = net(image)
            _, predicted = torch.max(output, 1)
            print(f'{path} -> {class_names[predicted.item()]}')
