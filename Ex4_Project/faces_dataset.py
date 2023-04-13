"""Custom faces dataset."""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        
        """Return a tuple containing an image and a label.
            Upon the transforms defined in utils.py, the image should be a Tensor image.
            The label is an integer: 0 for real images and 1 for fake / synthetic images.."""
            
        label =-1
        if index < len(self.real_image_names):
            label = 0 #real images
            img_name = os.path.join(self.root_path,"real",self.real_image_names[index])
        else:
            label = 1 #fake / synthetic images
            img_name = os.path.join(self.root_path,"fake",self.fake_image_names[index - len(self.real_image_names)])
        image = Image.open(img_name)
        
        if self.transform: # If the transform is not None, apply the transform on the image
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        return len(self.real_image_names) + len(self.fake_image_names)
