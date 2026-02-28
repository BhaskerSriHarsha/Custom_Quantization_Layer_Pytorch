import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model_def import base_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
base_m = base_model().to(DEVICE)

# Load the dataset
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std deviation of MNIST
])

# Download and load training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Download and load test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the loss function
loss_function = nn.CrossEntropyLoss()

# Load the optimizer
optimizer = optim.SGD(base_m.parameters(), lr=LEARNING_RATE)

def train_model():
    base_m.train() # Set model to training mode
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = base_m(images)
            loss = loss_function(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights

            total_loss += loss.item()

            # Print progress every 300 batches
            if (batch_idx + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# -------------------------------
# 6. Evaluation Loop
# -------------------------------
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


train_model()
test_model()
torch.save(base_m.state_dict(), 'base_saved_model.pth')
print('Model saved to disk!')