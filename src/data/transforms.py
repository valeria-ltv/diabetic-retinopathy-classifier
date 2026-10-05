from torchvision import transforms

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Get transforms for a given split.

    Args:
        split (str, optional): The split to get transforms for.
            Defaults to "train".
    Returns:
        transforms.Compose: The transforms to use.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=128),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize
        ])