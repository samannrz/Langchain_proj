import numpy as np
import matplotlib.pyplot as plt


def visualize_degradation(hr_volume, fr_index, crop_factor, noise_level, random_seed):
    # Select a frame
    hr_frame = hr_volume[fr_index]

    # Step 1: FFT to k-space
    frame_complex = hr_frame.astype(np.complex64)
    kspace = np.fft.fftshift(np.fft.fft2(frame_complex))

    # Save original k-space magnitude
    kspace_magnitude = np.log1p(np.abs(kspace))

    # Step 2: Crop k-space
    h, w = kspace.shape
    ch, cw = h // crop_factor, w // crop_factor
    start_h, start_w = (h - ch) // 2, (w - cw) // 2
    end_h, end_w = start_h + ch, start_w + cw
    cropped_kspace = kspace[start_h:end_h, start_w:end_w]

    # Step 3: Add noise
    np.random.seed(random_seed)
    if noise_level > 0:
        noise = noise_level * (np.random.randn(*cropped_kspace.shape) +
                               1j * np.random.randn(*cropped_kspace.shape))
        cropped_kspace += noise.astype(np.complex64)

    # Step 4: Inverse FFT
    degraded_frame = np.abs(np.fft.ifft2(np.fft.ifftshift(cropped_kspace)))

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(hr_frame, cmap='gray')
    axes[0].set_title('Original High-Res Frame')
    axes[0].axis('off')

    axes[1].imshow(kspace_magnitude, cmap='gray')
    axes[1].set_title('Original K-space (log scale)')
    axes[1].axis('off')

    axes[2].imshow(np.log1p(np.abs(cropped_kspace)), cmap='gray')
    axes[2].set_title(f'Cropped + Noisy K-space\nCrop Factor: {crop_factor}')
    axes[2].axis('off')

    axes[3].imshow(degraded_frame, cmap='gray')
    axes[3].set_title('Degraded Image (Low-Res)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
# visualize_degradation(hr_normalized, fr_index=4, crop_factor=4, noise_level=0.01, random_seed=42)
