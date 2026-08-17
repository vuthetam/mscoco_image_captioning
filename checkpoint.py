from accelerate import Accelerator
import torch


def save_checkpoint(path, encoder, decoder, optimizer, epoch, train_loss, best_val_loss, accelerator: Accelerator):
    #  chỉ cần 1 process ghi file
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save({
            "epoch": epoch,
            "train_loss": train_loss,
            "best_val_loss": best_val_loss,
            "encoder": accelerator.get_state_dict(encoder),
            "decoder": accelerator.get_state_dict(decoder),
            "optimizer": optimizer.state_dict(),
        }, path)


def load_checkpoint(path, encoder, decoder, device, optimizer=None):
    checkpoint = torch.load(path, map_location=device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    best_val_loss = checkpoint["best_val_loss"]

    return epoch, train_loss, best_val_loss
