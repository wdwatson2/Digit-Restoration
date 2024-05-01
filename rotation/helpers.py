import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def denormalize(tensor, mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]):
    # Assuming tensor is CxHxW
    # Reverse the normalization process
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add mean
    return tensor

def tensor_to_cv(tensor):
    # Ensure tensor is detached and on CPU
    numpy_image = tensor.detach().cpu().numpy()
    
    # Handle grayscale and color images differently
    if tensor.shape[0] == 1:  # Grayscale image, single channel
        # Remove channel dimension for grayscale, resulting in HxW
        numpy_image = numpy_image.squeeze()
    else:  # Color image, 3 channels
        # Convert from PyTorch's CHW format to OpenCV's HWC format
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
    
    # Convert from [0, 1] to [0, 255] and change to uint8
    cv_image = (numpy_image * 255).astype(np.uint8)
    return cv_image

def cv_to_tensor(cv_image):
    # Check if the image is grayscale (2D) or color (3D)
    if len(cv_image.shape) == 2:  # Grayscale image, 2D array
        # Add a channel dimension to make it HxWx1
        cv_image = cv_image[:, :, np.newaxis]
    
    # Scale pixel values from [0, 255] to [0, 1] and convert to float32
    numpy_image = cv_image.astype(np.float32) / 255.0
    
    # Rearrange from OpenCV's HWC format to PyTorch's CHW format
    tensor = torch.from_numpy(np.transpose(numpy_image, (2, 0, 1)))
    
    return tensor


def rotate_image(image,center, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Generate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def scale_image(image, scale):
    h, w = image.shape

    scaled_height, scaled_width = int(h * scale), int(w * scale)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))

    return scaled_image

class RotatedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Get the image and its label
        image_tensor, label = self.mnist_dataset[idx]

        # Convert the PyTorch tensor to an OpenCV image
        cv_image = tensor_to_cv(image_tensor)

        #Now lets create a rotation problem
        cent_horizontal, cent_vertical = np.random.uniform(10,15,size=2)
        rand_center = (cent_horizontal, cent_vertical)
        rand_rotation = np.random.randint(-180, 180)

        rotated_image = rotate_image(cv_image, rand_center, rand_rotation)

        #Convert back to tensors
        rotated_image = cv_to_tensor(rotated_image)
        rand_center = torch.tensor(rand_center, dtype=torch.float32)
        rand_rotation = torch.tensor([rand_rotation], dtype=torch.float32)

        return image_tensor, rotated_image, rand_center, rand_rotation
    
def generate_random_transformation_matrix(scale_range=(0.9, 1.1), rotation_range=(-60, 60),
                                          translation_range=(-1, 1), perspective_range=(-0.0005, 0.0005)):
    """
    Generates a random transformation matrix with controlled randomness.
    
    Parameters:
    - scale_range: Tuple (min, max) for random scaling factors.
    - rotation_range: Tuple (min, max) in degrees for random rotation angles.
    - translation_range: Tuple (min, max) for random translations.
    - perspective_range: Tuple (min, max) for random perspective distortion.
    
    Returns:
    - A 3x3 numpy array representing the random transformation matrix.
    """
    # Random scale factors for x and y axes
    sx, sy = np.random.uniform(scale_range[0], scale_range[1], 2)
    
    # Random rotation angle
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    # Random translations
    tx, ty = np.random.uniform(translation_range[0], translation_range[1], 2)
    
    # Random perspective distortion parameters
    g, h = np.random.uniform(perspective_range[0], perspective_range[1], 2)
    # g,h = (0,0)
    
    # Constructing the transformation matrix
    rotation_and_scale = np.array([[cos_theta * sx, -sin_theta * sy, 0],
                                   [sin_theta * sx, cos_theta * sy,  0],
                                   [0,               0,              1]])
    
    translation = np.array([[1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]])
    
    perspective = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [g, h, 1]])
    
    # Combine transformations: first apply rotation and scale, then translation, then perspective
    matrix = np.dot(np.dot(translation, rotation_and_scale), perspective)

    inverse = np.linalg.inv(matrix)

    
    return matrix, inverse, (g,h)
    

class PerspectiveChangeMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        # Get the image and its label
        image_tensor, label = self.mnist_dataset[idx]

        initial_image = image_tensor
        denormalized_initial = denormalize(image_tensor)

        # Convert the PyTorch tensor to an OpenCV image
        cv_image = tensor_to_cv(image_tensor)

        h,w,c = cv_image.shape

        perspective_range = (-0.0005, 0.0005)
        g2, h2 = np.random.uniform(perspective_range[0], perspective_range[1], 2)
        perspective = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [g2, h2, 1]])
        
        transformed_image = cv2.warpPerspective(cv_image, perspective, (h, w))

        # Convert the scaled OpenCV image back to a PyTorch tensor
        transformed_image = cv_to_tensor(transformed_image)  # Assuming you have a function to convert back

        inverse = np.linalg.inv(perspective)

        matrix = torch.tensor(list(perspective.flatten()), dtype=torch.float32)

        inverse = torch.tensor(list(inverse.flatten()), dtype=torch.float32)

        inv_g2, inv_h2 = inverse[6], inverse[7]

        g2 = torch.tensor(g2, dtype=torch.float32)
        h2 = torch.tensor(h2, dtype=torch.float32)

        return image_tensor, transformed_image, matrix, inverse, (g2,h2), (inv_g2,inv_h2)
        
    
def imshow(img):
    img = img.numpy()  # Convert tensor to numpy array
    img = np.transpose(img, (1, 2, 0))  # Change the order from (C, H, W) to (H, W, C)
    plt.imshow(img, cmap='gray')  # Display the image
    plt.axis('off')  # Turn off the axis

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, mode='auto'):
        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum change in monitored quantity to qualify as improvement
        self.verbose = verbose  # Whether to print information about the early stopping
        self.mode = mode  # 'auto', 'min', or 'max'

        self.counter = 0  # Counter to keep track of epochs without improvement
        self.best_score = None  # Best validation score
        self.early_stop = False  # Flag to indicate if training should stop

        if self.mode not in ['auto', 'min', 'max']:
            raise ValueError("Mode must be one of 'auto', 'min', or 'max'.")

        if self.mode == 'min':
            self.delta *= -1  # For 'min' mode, reverse the delta direction

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif ((val_score - self.best_score) <= self.delta and self.mode != 'min') or ((val_score - self.best_score) >= self.delta and self.mode == 'min'):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def custom_loss(outputs, targets, weights:tuple):
    # Extract predicted and true centers and angles
    pred_centers, pred_angles = outputs[:, :2], outputs[:, 2]
    true_centers, true_angles = targets[:, :2], targets[:, 2]
    
    # Calculate Euclidean distance for centers
    center_loss = torch.sqrt(torch.sum((pred_centers - true_centers) ** 2, dim=1))
    
    # Calculate angle loss considering cyclical nature
    angle_diff = torch.abs(true_angles - pred_angles) % 360  # Modulo to consider cyclical effect
    angle_loss = torch.min(angle_diff, 360 - angle_diff)  # Choose smaller difference due to cycle
    
    center_weight, angle_weight = weights
    # Combine losses
    loss = torch.mean(center_weight*center_loss + angle_weight*angle_loss)  # Mean to average over batch
    return loss