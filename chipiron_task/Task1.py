# Task 1
# TO DO - write a function to normalize an MRI volume by its max
import numpy as np
def normalize_volume(vol: np.ndarray):
 return (vol-vol.min())/(vol.max()-vol.min())

# TO DO - write a function to calculate the psnr of an image
def calculate_psnr_image(image: np.ndarray, ref: np.ndarray):
  mse = np.mean((image - ref) ** 2)
  if mse == 0:
      return float('inf')  # Perfect match
  max_pixel = 1.0  # Since images are normalized
  psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
  return psnr

# TO DO - write a function to resize an image (you can use bicubic interpolation for example)
import cv2
def resize_frame(frame: np.ndarray, target_size: tuple):
  resized_frame = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
  return resized_frame