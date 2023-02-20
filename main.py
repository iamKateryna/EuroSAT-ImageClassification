import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet50
from transformers import ViTForImageClassification, ViTFeatureExtractor

import argparse
import wandb


from eurosat import EuroSAT
from transformereurosat import TransformerEuroSAT
from train import train
from evaluate import evaluate

np.random.seed(42)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
FOLDER_PATH = './2750'  # path to folder with data
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
VIT_CHECKPOINT_PATH = 'google/vit-base-patch16-224-in21k'

PROJECT_NAME = 'eurosat-image-classification'
CHECKPOINT_PATH = 'checkpoints/'


def prepare_data(model_name):
    dataset = ImageFolder(FOLDER_PATH)

    indices = list(range(len(dataset)))

    train_split = int(TRAIN_SIZE * len(dataset))
    val_split = int((TRAIN_SIZE+VAL_SIZE) * len(dataset))
    np.random.shuffle(indices)

    train_data = Subset(dataset, indices=indices[:train_split])
    val_data = Subset(dataset, indices=indices[train_split:val_split])
    test_data = Subset(dataset, indices=indices[val_split:])

    # data transformation for different models
    if model_name == 'resnet50':

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        train_data = EuroSAT(train_data, train_transform)
        val_data = EuroSAT(val_data, test_transform)
        test_data = EuroSAT(test_data, test_transform)

    else:

        feature_extractor = ViTFeatureExtractor.from_pretrained(
            VIT_CHECKPOINT_PATH)

        train_data = TransformerEuroSAT(train_data, feature_extractor)
        val_data = TransformerEuroSAT(val_data, feature_extractor)
        test_data = TransformerEuroSAT(test_data, feature_extractor)

    return train_data, val_data, test_data


def get_model(model_name, device, number_of_classes):

    if model_name == 'resnet50':

        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)

    elif model_name == 'vit':

        model = ViTForImageClassification.from_pretrained(
            VIT_CHECKPOINT_PATH, num_labels=number_of_classes)

    return model.to(device)


def main(args):

    wandb.init(project=PROJECT_NAME)
    model_path = f'{CHECKPOINT_PATH}/{args.model_name}-model.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, val_data, test_data = prepare_data(args.model_name)

    print(f'Train/val/test sizes: {len(train_data)}/{len(val_data)}/{len(test_data)}')

    number_of_classes = len(train_data.dataset.dataset.classes)
    model = get_model(args.model_name, device, number_of_classes)

    dataloaders = {
        'train': DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True),
        'val': DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True),
        'test': DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2)

    best_loss = np.inf

    # Train and test over n_epochs
    for epoch in tqdm(range(args.num_epochs)):
        print(f'Epoch {epoch+1}')
        train(model, device, dataloaders['train'],
              number_of_classes, criterion, optimizer)
        val_loss, _, _ = evaluate(
            model, device, dataloaders['val'], number_of_classes, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)

            if wandb.run is not None:
                wandb.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    main(args)
