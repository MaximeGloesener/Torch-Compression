"""
Example of how to use the optimize function in main.py on CIFAR10 dataset 
"""

from main import optimize, evaluate
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader
from models.model import resnet20 
import torch 


# dataloader for cifar10
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    }
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(root="data/cifar10", train=(split == "train"), download=True, transform=transforms[split])
dataloader = {}
for split in ['train', 'test']:
    dataloader[split] = DataLoader(dataset[split], batch_size=128, shuffle=(split == 'train'), num_workers=0, pin_memory=True)


# model
model = resnet20(num_classes=10)
model.load_state_dict(torch.load("models/cifar10_resnet20.pt"))
model = model.cuda()

# evaluate 
acc, loss = evaluate(model, dataloader['test'])
print(f"Accuracy: {acc}, Loss: {loss}")

# optimize
example_input = torch.randn(1, 3, 32, 32).cuda()
num_classes = 10
model = optimize(model, dataloader['train'], dataloader['test'], epochs=3, lr=0.1, example_input=example_input, num_classes=num_classes)


