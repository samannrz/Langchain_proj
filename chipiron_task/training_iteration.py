# Training iteration
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from Task1 import calculate_psnr_image
import numpy as np
def train_iter(
    epoch: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    batch_size: int,
) -> tuple[float, float, np.ndarray | None]:
    """Perform one epoch of training and validation for the model.

    Args:
        epoch: The current epoch number.
        model: The model to train and evaluate.
        device: The device to use for computation.
        train_dataloader: Dataloader for the training dataset.
        val_dataloader: Dataloader for the validation dataset.
        criterion: Loss function to compute training loss.
        optimizer: Optimizer for updating model weights.
        n_epochs: Total number of epochs for training.
        batch_size: Batch size for training.

    Returns:
        A tuple containing:
        - Average training loss of the epoch.
        - Average validation PSNR of the epoch.
        - A predicted validation image.
    """
    # Training phase
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    epoch_loss = 0
    n_train_samples = len(train_dataloader.dataset)

    with tqdm(total=(n_train_samples - n_train_samples % batch_size), ncols=80, leave=False) as t:
        t.set_description(f"epoch {epoch}/{n_epochs - 1} - training")

        for data in train_dataloader:
            train_inputs, train_labels = data

            
            train_inputs = train_inputs.to(device)
            train_labels = train_labels.to(device)

            with torch.autocast(device_type="cuda"):
                train_preds = model(train_inputs)
                loss = criterion(train_preds, train_labels)

            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t.set_postfix(loss=f"{epoch_loss/len(train_inputs):.6f}")
            t.update(len(train_inputs))

    # Val phase: quantitative (psnr) and qualitative (render prediction for 1 sample)
    model.eval()
    epoch_psnr = 0
    epoch_ssim = 0
    n_val_samples = len(val_dataloader.dataset)
    with tqdm(total=n_val_samples, ncols=80, leave=False) as t:
        t.set_description(f"epoch {epoch}/{n_epochs - 1} - validation")
        for data in val_dataloader:
            val_input, val_label = data

            val_input = val_input.to(device)
            val_label = val_label.to(device)

            with torch.no_grad(), torch.autocast(device_type="cuda"):
                val_pred = model(val_input)

            # Compute psnr
            frame_val_pred = val_pred.squeeze().cpu().numpy()
            frame_val_label = val_label.squeeze().cpu().numpy()
            sample_psnr = calculate_psnr_image(frame_val_pred, frame_val_label)
            epoch_psnr += sample_psnr
            t.update(1)
    epoch_psnr = epoch_psnr / len(val_dataloader)

    # For the last val sample, store prediction
    val_pred = val_pred.squeeze().cpu().numpy()

    return epoch_loss / len(train_inputs), epoch_psnr, epoch_ssim, val_pred
