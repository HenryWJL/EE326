import cv2
import numpy as np
from typing import Optional
from datetime import datetime
from scipy.interpolate import interp2d
from scipy.interpolate import NearestNDInterpolator


def nearest_neighbor(
    input_file: Optional[str],
    dim: Optional[tuple]
    ) -> np.ndarray:
    """
    Params:
        input_file: the file name of the original image
        
        dim: the desired shape of the interpolated image
        
    Returns:
        the interpolated image
        
    """
    img = cv2.imread(input_file, 0)
    h, w = img.shape
    h_new, w_new = dim
    assert isinstance(h_new, int) and isinstance(w_new, int), "Height and width must be integers!"
    assert h_new != 0 and w_new != 0, "Both height and width must be nonzero!"
    
    img_new = np.zeros((h_new, w_new), dtype=np.uint8)
    for y_new in range(h_new):
        for x_new in range(w_new):
            ### project (x_new, y_new) to (x, y), where (x_new, y_new)
            #   are the pixel coordinates of the interpolated images and
            #   (x, y) are the pixel coordinates of the original images  
            y = (y_new / h_new) * h
            x = (x_new / w_new) * w
            ### find the nearest coordinates
            y = round(y)
            x = round(x)
            ### address out-of-bound error
            if y == h:
                y -= 1
            if x == w:
                x -= 1
            ### interpolation
            img_new[y_new][x_new] = img[y][x]
            
    return np.array(img_new, dtype=np.uint8)


def bilinear(
    input_file: Optional[str],
    dim: Optional[tuple]
    ) -> np.ndarray:
    """
    Params:
        input_file: the file name of the original image
        
        dim: the desired shape of the interpolated image
        
    Returns:
        the interpolated image
        
    """
    img = cv2.imread(input_file, 0)
    h, w = img.shape
    h_new, w_new = dim
    assert isinstance(h_new, int) and isinstance(w_new, int), "Height and width must be integers!"
    assert h_new != 0 and w_new != 0, "Both height and width must be nonzero!"
    
    img_new = np.zeros((h_new, w_new), dtype=np.uint8)
    for y_new in range(h_new):
        for x_new in range(w_new):
            ### project (x_new, y_new) to (x, y), where (x_new, y_new)
            #   are the pixel coordinates of the interpolated images and
            #   (x, y) are the pixel coordinates of the original images.  
            y = (y_new / h_new) * h
            x = (x_new / w_new) * w
            ### address out-of-bound error
            if int(y) == (h - 1):
                y -= 1
            if int(x) == (w - 1):
                x -= 1
            ### find neighbors
            y_0 = int(y)
            x_0 = int(x)
            y_1 = y_0 + 1
            x_1 = x_0 + 1
            ### compute weights
            w_00 = (y_1 - y) * (x_1 - x)
            w_10 = (y_1 - y) * (x - x_0)
            w_01 = (y - y_0) * (x_1 - x)
            w_11 = (y - y_0) * (x - x_0)
            ### interpolation
            img_new[y_new][x_new] = w_00 * img[y_0][x_0] + w_10 * img[y_0][x_1] + \
                                    w_01 * img[y_1][x_0] + w_11 * img[y_1][x_1]
            
    return np.array(img_new, dtype=np.uint8)


def bicubic(
    input_file: Optional[str],
    dim: Optional[tuple]
    ) -> np.ndarray:
    """
    Params:
        input_file: the file name of the original image
        
        dim: the desired shape of the interpolated image
        
    Returns:
        the interpolated image
        
    """
    img = cv2.imread(input_file, 0)
    h, w = img.shape
    h_new, w_new = dim
    assert isinstance(h_new, int) and isinstance(w_new, int), "Height and width must be integers!"
    assert h_new != 0 and w_new != 0, "Both height and width must be nonzero!"
    
    y = (np.arange(h_new) / h_new) * h
    x = (np.arange(w_new) / w_new) * w
    
    interp = interp2d(np.arange(w), np.arange(h), img, 'cubic')
    img_new = interp(x, y)
            
    return np.array(img_new, dtype=np.uint8)


def interpolate(
    img: Optional[np.ndarray],
    dim: Optional[tuple],
    option: Optional[str] = "nearest"
    ) -> np.ndarray:
    """
    Params:
        img: the original image
        
        dim: the desired shape of the interpolated image
        
        option: "nearest", "linear", "cubic", "quintic"
        
    Returns:
        the interpolated image
        
    """
    h, w = img.shape
    h_new, w_new = dim
    assert isinstance(h_new, int) and isinstance(w_new, int), "Height and width must be integers!"
    assert h_new != 0 and w_new != 0, "Both height and width must be nonzero!"
    
    y = (np.arange(h_new) / h_new) * h
    x = (np.arange(w_new) / w_new) * w
    
    if option == "nearest":
        H, W = np.meshgrid(np.arange(h), np.arange(w))
        X, Y = np.meshgrid(x, y)
        coordinates = np.concatenate((H[:, :, None], W[:, :, None]), axis=-1).reshape(-1, 2)
        img = img.reshape(-1)
        
        start_time = datetime.now()
        for i in range(100):
            interp = NearestNDInterpolator(coordinates, img)
            img_new = interp(X, Y)
        end_time = datetime.now()
        duration = round((end_time - start_time).microseconds / 10**6, 3)
        
    else:
        W, H = np.arange(w), np.arange(h)
        
        start_time = datetime.now()
        for i in range(100):
            interp = interp2d(W, H, img, option)
            img_new = interp(x, y)
        end_time = datetime.now()
        duration = round((end_time - start_time).microseconds / 10**6, 3)
            
    return np.array(img_new, dtype=np.uint8), duration
