from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])