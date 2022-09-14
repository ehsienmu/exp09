import os
import sys
import random
from vision import VisionDataset
from PIL import Image
import cv2
import os.path
import numpy as np
import torch
from torchvision import transforms

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def opencv_loader(path, colorSpace='RGB'):
    image = cv2.imread(str(path))
    if colorSpace == "YCrCb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        print(image.shape)
    elif colorSpace == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def default_loader(path, backend='opencv', colorSpace='RGB'):
    from torchvision import get_image_backend
    if backend == 'opencv':
        return opencv_loader(path, colorSpace=colorSpace)
    else:
        raise NotImplementedError

class DatasetFolder(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None,
                 backend='opencv', aggregate=False):
        super(DatasetFolder, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.backend=backend
        self.aggregate = aggregate

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.backend == 'opencv':
            images = self.loader(path, backend='opencv', colorSpace='RGB')
            
        if self.transform is not None:
            PIL_image = Image.fromarray(np.uint8(images))
            return self.transform(PIL_image) , target

    def __len__(self):
        return len(self.samples)

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, backend='opencv', aggregate=False):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          backend=backend,
                                          aggregate=aggregate)
        self.imgs = self.samples