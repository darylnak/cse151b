import torch
import pandas as pd # to help with getitem
from skimage import io, transform # help with image proccessing
from torch.utils.data import Dataset
import os
import numpy as np

# Resources: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class
class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self, root, file_path, transform=None):
        self.image_frame = pd.read_csv(os.getcwd() + "/birds_dataset/" + 
                                       file_path, sep=" ", header=None)
        self.root_dir = root
        self.transform = transform
        self.targets = self.image_frame.iloc[:, 1].to_numpy()

    def __len__(self):
        return len(self.image_frame)

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir,
                                  self.image_frame.iloc[idx, 0])
        image = io.imread(image_path)

        # check for grayscale image
        # convert to fake RGB (3 channel grayscale)
        if len(image.shape) == 2:
            image = np.dstack( (np.dstack((image, image)), image) )

        label = self.image_frame.iloc[idx, 1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# This class has been adapted from the following pytorch tutorial:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # check for valid input
        self.output_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int): # check for int
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}


# This class has been adapted from the following pytorch tutorial:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int) # check for valid input
        if isinstance(output_size, int): # check for int
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2 # check valid tuple
            self.output_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # randomly select top and left offset
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # crop image and retain output_size size
        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors (faster)."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0 ,1))
        return {'image': torch.from_numpy(image), 
                'label': label}

class Normalize(object):
    def __init__(self, mean, std):
        assert isinstance(mean, list)
        assert isinstance(std, list)

        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Normalize with mean and standard deviation
        # Resource:
        # https://stackoverflow.com/questions/53987906/how-to-multiply-row-wise-by-scalar-in-pytorch
        image = image - self.mean[:, None, None]
        image = image / self.std[:, None, None]

        return {'image': image, 'label': label}