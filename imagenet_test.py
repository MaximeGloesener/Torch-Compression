import torchvision
import torchvision.transforms as transforms

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download the ImageNet validation set
imagenet_val = torchvision.datasets.ImageNet(
    root='./data',
    split='val',
    download=True,
    transform=transform
)

from torch.utils.data import DataLoader

val_loader = DataLoader(imagenet_val, batch_size=32, shuffle=False, num_workers=4)