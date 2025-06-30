# Task 2 - degradation
# TO DO - write a function to degrade an MRI volume e.g by (i) cropping its kspace and (ii) adding noise
import numpy as np

def degrade_volume(vol: np.ndarray, crop_factor: int, noise_level: float, random_seed: int):
    """
    Degrade an MRI volume by (1) cropping its k-space and (2) adding noise.

    Args:
        vol (np.ndarray): Input volume of shape (N, H, W)
        crop_factor (int): Factor by which to crop the k-space (e.g., 2 = 1/2 in each dimension)
        noise_level (float): Standard deviation of Gaussian noise to add
        random_seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Degraded volume of shape (N, H // crop_factor, W // crop_factor)
    """
    np.random.seed(random_seed)
    degraded_vol = []

    for frame in vol:
        # Step 1: FFT to k-space
        frame_complex = frame.astype(np.complex64)
        kspace = np.fft.fftshift(np.fft.fft2(frame_complex))

        # Step 2: Center crop k-space
        h, w = kspace.shape
        ch, cw = h // crop_factor, w // crop_factor
        start_h, start_w = (h - ch) // 2, (w - cw) // 2
        end_h, end_w = start_h + ch, start_w + cw
        cropped_kspace = kspace[start_h:end_h, start_w:end_w]

        # Step 3: Add complex Gaussian noise (on the cropped region)
        if noise_level > 0:
            noise = noise_level * (np.random.randn(*cropped_kspace.shape) +
                                   1j * np.random.randn(*cropped_kspace.shape))
            cropped_kspace += noise.astype(np.complex64)

        # Step 4: Inverse FFT back to image space
        degraded_frame = np.abs(np.fft.ifft2(np.fft.ifftshift(cropped_kspace)))
        degraded_vol.append(degraded_frame)

    return np.stack(degraded_vol)