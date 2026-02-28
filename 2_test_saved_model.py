import torch
from base_model_def import base_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
DEVICE = "cuda:0"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std deviation of MNIST
])
BATCH_SIZE=32

# Download and load test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

base_m = base_model().to(DEVICE)
base_m.load_state_dict(torch.load("base_saved_model.pth", map_location=DEVICE, weights_only=True))

base_m.eval()

def test_model():
    base_m.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient computation for testing (saves memory & compute)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = base_m(images)
            
            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy on the 10,000 test images: {accuracy:.2f}%")

test_model()

# Quantize the model now
