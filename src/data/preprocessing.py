import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """
    Reads an image from disk and converts its color space for proper display.

    Args:
        path (str): Full path to the image file.
    Returns:
        numpy.ndarray: Image array in RGB format.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def crop_black_borders(img: np.ndarray, tol: int = 10) -> np.ndarray:
    """
    Crops black borders around an image.

    Args:
        img (np.ndarray): Image array in RGB format.
        tol (int, optional): Tolerance threshold for black pixels. Defaults to 10.

    Returns:
        np.ndarray: Cropped image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax + 1, xmin:xmax + 1]

def resize_with_padding(img: np.ndarray, target_size: tuple[int, int],
                        pad_color: tuple[int] = (0, 0, 0)) -> np.ndarray:
    """
    Resizes image to target size and pads it saving aspect ratio.

    Args:
        img (np.ndarray): Image array in RGB format.
        target_size (int): Target size in pixels.
        pad_color (tuple[int], optional): Padding color. Defaults to (0, 0, 0).

    Returns:
        np.ndarray: Resized image.
    """
    height, width = img.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))

    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - pad_top - new_height
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - pad_left - new_width

    return cv2.copyMakeBorder(
        resized_img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )

def resize(img: np.ndarray, target_size: int, method: str) -> np.ndarray:
    """
    Resizes an image to the target size, either preserving the aspect ratio or stretching.

    Args:
        image (np.ndarray): Image array in RGB format.
        target_size (int): Target size in pixels.
        method (str): Resize method ('stretch' or 'preserve_aspect_ratio').
    Returns:
        np.ndarray: Resized image.
    """
    if method == 'preserve_aspect_ratio':
        img = resize_with_padding(img, target_size=(target_size, target_size))
    elif method == 'stretch':
        img = cv2.resize(img, (target_size, target_size))
    return img

def ben_preprocess(img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Applies Ben's preprocessing technique.

    Args:
        image (np.ndarray): Image array in RGB format.
        target_size (int): Target size in pixels.
    Returns:
        np.ndarray: Preprocessed image.
    """
    return cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), target_size / 30.0), -4, 128
    )

def remove_border(img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Removes the boundary artifact of an image by applying a circular mask.

    Args:
        img (np.ndarray): Image array in RGB format.
        target_size (int): Target size in pixels.
    Returns:
        np.ndarray: Image with border removed.
    """
    # Circle mask (90% of radius)
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    center = (target_size // 2, target_size // 2)
    radius = int((target_size / 2) * 0.9)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    img[mask == 0] = 128
    return img

def preprocess_pipeline(img_path: str, target_size: int = 384) -> np.ndarray:
    """
    Preprocesses image.

    Args:
        img_path (str): Path to the image file to preprocess.
        target_size (int, optional): Target size of a square image in pixels. Defaults to 384.
    Returns:
        np.ndarray: Preprocessed image in RGB format.
    """
    img = load_image(img_path)
    img = crop_black_borders(img)
    img = resize(img, target_size=target_size, method="stretch")
    img = ben_preprocess(img, target_size=target_size)
    img = remove_border(img, target_size=target_size)
    return img