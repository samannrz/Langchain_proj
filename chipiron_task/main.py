# Imports
import copy
import os
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from functions import *
from config import *

# TEST FUNCTIONS
def Test_functions(config):
    import h5py
    data_dir = config["val_folder"][0]

    all_files = os.listdir(data_dir) # List all files in the directory
    file_name = all_files[0] # select a file
    file_path = os.path.join(data_dir, file_name)

    hf = h5py.File(file_path, 'r')
    volume_kspace = hf['kspace'][()]
    print(volume_kspace.shape)

    with h5py.File(file_path, 'r') as hf:
              # Load the reconstruction_rss dataset
              volume_rss = hf['reconstruction_rss'][()]
    print((volume_rss.shape))
    volume_plot(volume_rss)
    volume_rss=normalize_volume(volume_rss)

from data_class import *
# Data initialization
train_dataset = MRDataset2D(
 config["train_folder"],
 crop_factor,
 config["noise_level"],
 config["random_seed"],
)

val_dataset = MRDataset2D(
 config["val_folder"],
 crop_factor,
 config["noise_level"],
 config["random_seed"],
)

train_dataloader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1)

###################
# Render a val sample with low-res / high-res images
lowres_val_sample, highres_val_sample = val_dataset[len(val_dataset) - 1]

lowres_val_sample = lowres_val_sample.squeeze().cpu().numpy()
highres_val_sample = highres_val_sample.squeeze().cpu().numpy()
# Create a side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(lowres_val_sample, cmap="gray")
axes[0].set_title("Low-res input")
axes[0].axis("off")

axes[1].imshow(highres_val_sample, cmap="gray")
axes[1].set_title("High-res ground truth")
axes[1].axis("off")

plt.tight_layout()
plt.show()
####################
# Compute psnr for val dataset without training
initial_psnr = 0

with tqdm(total=len(val_dataloader), ncols=80, leave=False) as t:
    t.set_description("Computing initial psnr on validation set")
    for data in val_dataloader:
        val_input, val_label = data

        # Compute psnr
        frame_val_input = val_input.squeeze().cpu().numpy()
        frame_val_label = val_label.squeeze().cpu().numpy()
        # Resize input to same size as label
        frame_val_input = resize_frame(
        frame_val_input,
        target_size=(frame_val_label.shape[0], frame_val_label.shape[1]),
        )
        sample_psnr = calculate_psnr_image(frame_val_input, frame_val_label)
        initial_psnr += sample_psnr
        t.update(1)
initial_psnr = initial_psnr / len(val_dataloader)
print(f"Initial psnr = {initial_psnr:.2f}")
########################
from model_class import *
##################
config_model = {"growth_rate":32, "n_blocks":4, "n_layers":4}
config_training = {"n_epochs":5,"lr":0.0001,"patience":5}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRDenseNet2D(
 train_dataset[0][0].shape[0],
 config_model["growth_rate"],
 2,
 config_model["n_blocks"],
 config_model["n_layers"],
).to(device)

optimizer = optim.Adam(model.parameters(), config_training["lr"])
best_weights = copy.deepcopy(model.state_dict())
#######################
from Task3 import *
#######################
if __name__ == "__main__":
    from training_iteration import train_iter
    import copy
    from matplotlib import pyplot as plt

    results = {"train_loss": [], "val_psnrs": []}
    best_psnr = -float("inf")
    no_progress = 0
    best_epoch = -1
    best_weights = None

    for epoch in range(config_training["n_epochs"]):
        # Decrease learning rate with epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = config_training["lr"] * (
                0.1 ** (epoch // int(config_training["n_epochs"] * 0.8))
            )

        # Run one epoch of training + validation
        epoch_train_loss, epoch_val_psnr, _, epoch_val_image = train_iter(
            epoch,
            model,
            device,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            config_training["n_epochs"],
            config["batch_size"],
        )

        # Log epoch
        results["train_loss"].append(epoch_train_loss)
        results["val_psnrs"].append(epoch_val_psnr)
        fig_epoch = plt.figure(figsize=(9, 9))
        plt.imshow(epoch_val_image, cmap="gray")
        plt.title(f"Pred high-res (epoch = {epoch})")
        plt.axis("off")
        plt.show()

        if round(epoch_val_psnr, 5) > round(best_psnr, 5):
            # Save the best ckpt according to val psnr
            no_progress = 0
            best_epoch = epoch
            best_psnr = epoch_val_psnr
            best_weights = copy.deepcopy(model.state_dict())
        else:
            no_progress += 1
            if no_progress == config_training["patience"]:
                print(
                    f"Early stopping, no progress reached since {config_training['patience']} epochs"
                )
                break# Save best model
model_path = "Output/MSE_10/model.pt"
torch.save(best_weights, model_path)

plt.figure(figsize=(8, 6))
plt.plot(results["train_loss"], label="Training Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
loss_plot_path = "Output/MSE_TV_10/training_loss_curve.png"
plt.savefig(loss_plot_path)
print(f"Training loss plot saved to: {loss_plot_path}")