import torch
import numpy as np
import matplotlib.pyplot as plt
from model_class import SRDenseNet2D
from config import crop_factor, noise_level, random_seed
from Task1 import normalize_volume, calculate_psnr_image, resize_frame
from Task2 import degrade_volume
import h5py

# Load low-res and high-res frame as you already have
infererence_file_path = '2022092502_T102.h5'
fr_index = 7
hf = h5py.File(infererence_file_path, 'r')
hr_volume = np.abs(hf["reconstruction_rss"][:])
hr_normalized = normalize_volume(hr_volume)
hr_frame = hr_normalized[fr_index]

lr_volume = degrade_volume(hr_normalized, crop_factor, noise_level, random_seed)
lr_frame = lr_volume[fr_index]

print(f"Low-res frame shape: {lr_frame.shape}")

# Prepare the model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRDenseNet2D(
    n_slices=1,         # Assuming grayscale single slice input
    growth_rate=32,
    sr_factor=2,        # Same as in training
    n_blocks=4,
    n_layers=4
).to(device)

# Load trained weights
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# Preprocess low-res frame for model input
# - model expects input as torch tensor with shape (batch_size, channels, H, W)
input_tensor = torch.from_numpy(lr_frame).unsqueeze(0).unsqueeze(0).float().to(device)

# Forward pass through the model
with torch.no_grad():
    output = model(input_tensor)

# Output shape should be (1, 1, H_sr, W_sr), convert to numpy and squeeze
output_img = output.squeeze().cpu().numpy()
resized_input = resize_frame(
        lr_frame,
        target_size=(hr_frame.shape[0], hr_frame.shape[1]),
        )
print(calculate_psnr_image(resized_input,hr_frame))
print(calculate_psnr_image(output_img,hr_frame))
# Display the input low-res, output super-res, and high-res ground truth side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(lr_frame, cmap='gray')
axes[0].set_title("Low-res input")
axes[0].axis('off')

axes[1].imshow(output_img, cmap='gray')
axes[1].set_title("Super-res output")
axes[1].axis('off')

axes[2].imshow(hr_frame, cmap='gray')
axes[2].set_title("High-res ground truth")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('output.png',dpi=300)

plt.show()
