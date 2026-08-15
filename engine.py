from torch import nn
from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from torch.optim import Optimizer


def train_one_epoch(encoder: nn.Module, decoder: nn.Module, dataloader, optimizer: Optimizer, criterion, accelerator: Accelerator, max_norm = 1.0):
    encoder.train() 
    decoder.train()

    running_loss: float = 0
    pbar = tqdm(dataloader, desc="Training", leave=False, disable=not accelerator.is_local_main_process)

    params = list(filter(
        lambda p: p.requires_grad,
        list(encoder.parameters()) + list(decoder.parameters())
    ))

    for images, captions, attn_masks in pbar:
        # chuẩn bị dữ liệu cho teacher forcing
        inputs = captions[:, :-1]
        pad_masks = attn_masks[:, :-1] 
        targets = captions[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        memory = encoder(images)
        logits = decoder(memory, inputs, pad_masks)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),    # [B x seq_length, vocab_size]
            targets.reshape(-1)                     # [B x seq_length]
        )
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(params, max_norm)

        optimizer.step()

        train_loss = accelerator.reduce(loss.detach(),reduction="mean").item()
        running_loss += train_loss

        pbar.set_postfix({"Train loss": f"{train_loss:.4f}"})

    return running_loss / len(dataloader)


def evaluate_one_epoch(encoder: nn.Module, decoder: nn.Module, dataloader, criterion, accelerator: Accelerator):
    encoder.eval()
    decoder.eval()

    running_loss = 0.0

    with torch.no_grad():
        for images, captions, attn_masks in dataloader:
            inputs = captions[:, :-1]
            pad_masks = attn_masks[:, :-1]
            targets = captions[:, 1:]

            memory = encoder(images)
            logits = decoder(memory, inputs, pad_masks)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            val_loss = accelerator.reduce(loss.detach(),reduction="mean").item()
            running_loss += val_loss

        return running_loss / len(dataloader)