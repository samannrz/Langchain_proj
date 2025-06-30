# Initialization
import torch
noise_level = 5
random_seed = 42
batch_size = 4
crop_factor = 2
config = {"train_folder": ['/Users/saman/Documents/chipiron_task/input/m4raw_multicoil_train'], "val_folder":['/Users/saman/Documents/chipiron_task/input/m4raw_multicoil_val'], "noise_level":noise_level, "random_seed":random_seed, "batch_size":batch_size}
torch.manual_seed(config["random_seed"])
device = torch.device("cuda")
## Metrics
best_epoch = 0
best_psnr = 0.0
no_progress = 0

