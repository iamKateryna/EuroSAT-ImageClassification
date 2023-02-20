import torch
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_average_precision
from transformers.modeling_outputs import ImageClassifierOutput
import wandb

def evaluate(model, device, dataloader, number_of_classes, criterion):
    model.eval()

    running_loss = 0.0
    predictions, targets = [], []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            #transformer model outputs an ImageClassifierOutput object
            if isinstance(outputs, ImageClassifierOutput):
                outputs = outputs.logits

            loss = criterion(outputs, labels) 

        running_loss += loss.item() * inputs.size(0)
        predictions.append(outputs.cpu())
        targets.append(labels.cpu())

    predictions = torch.cat(predictions, 0)
    targets = torch.cat(targets, 0)

    # Calculate epoch loss and accuracy
    epoch_f1 = multiclass_f1_score(predictions, targets, num_classes=number_of_classes)
    epoch_auprc = multiclass_average_precision(predictions, targets, num_classes=number_of_classes)
    epoch_loss = running_loss / len(dataloader.dataset)

    if wandb.run is not None:
        wandb.log({'val_loss': epoch_loss, 'val_f1': epoch_f1, 'val_auprc': epoch_auprc})

    return epoch_loss, epoch_f1, epoch_auprc