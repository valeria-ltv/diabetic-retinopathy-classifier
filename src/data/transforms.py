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
    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])