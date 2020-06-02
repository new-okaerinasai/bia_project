import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import tqdm
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision.transforms import (Compose, RandomAffine, RandomPerspective,
                                    RandomHorizontalFlip, RandomVerticalFlip,
                                    RandomResizedCrop, Normalize, ToTensor,
                                    ColorJitter, Resize)
from PIL import Image
from sklearn.metrics import confusion_matrix
import json
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

IMG_SIZE = 512


def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img, sigmaX=10):
    """
    Create circular crop around image centre    
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = crop_image_from_gray(img)
    height, width, depth = img.shape

    x = int(width/2)
    y = int(height/2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(
        img, (0, 0), IMG_SIZE / 10), -4, 128)
    return img


class RetinaDataset(Dataset):
    def __init__(self, images_dir: str, csv_path=None, mode='train'):
        if mode == "train":
            self.markup = pd.read_csv(csv_path)
        self.images_ids = []
        self.mode = mode
        self.images_dir = images_dir
        self.images = []
        for idx in tqdm.tqdm(os.listdir(images_dir)):
            self.images_ids.append(idx)
            self.images.append(circle_crop(
                cv2.imread(os.path.join(self.images_dir, idx))))

    def __getitem__(self, i: int):
        idx = self.images_ids[i]
        res = (idx,)
        image = self.images[i]
        res += (image,)
        if self.mode == "train":
            res += (self.markup[self.markup['id_code']
                                == idx.split(".")[0]].values[0][1],)
        return res

    def __len__(self):
        return len(self.images_ids)


class RetinaSubsetDataset(Dataset):
    def __init__(self, original, subset, transform):
        self.dataset = original
        self.subset = np.sort(subset)
        self.transform = transform

    def __getitem__(self, i):
        idx = self.subset[i]
        id, image, label = self.dataset[idx]
        return (id, self.transform(Image.fromarray(image)), label)

    def __len__(self):
        return len(self.subset)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp),
        "npv": tn / (tn + fn),
        "accuracy": (tp + tn) / (tp + tn + fp + fn)
    }


def evaluate(model, val_dataloader, criterion, device, thr=0.2):
    """
    Evaluate model. This function returns validation loss and accuracy.
    :param model: model to evaluate
    :param dataloader: dataloader representing the validation data
    :param callable criterion: the loss function
    :return: None
    """
    # Validation
    model = model.to(device).eval()
    all_predictions = []
    all_losses = []
    all_dr = []
    all_dr_pred = []
    with torch.no_grad():
        for image_id, images, labels in tqdm.tqdm(val_dataloader):
            images, labels = images.to(device), labels.to(device).long()
            logits = model.forward(images)

            labels = labels.cpu().detach().numpy()
            logits = F.softmax(logits, dim=-1).cpu().detach().numpy()
            predictions = logits.argmax(1)
            all_predictions.append((predictions == labels))
            all_losses.append(criterion(torch.from_numpy(
                logits), torch.from_numpy(labels)).item())
            any_dr = (labels != 0)
            any_dr_pred = 1 - (logits[:, 0] > thr)
            all_dr.append(any_dr)
            all_dr_pred.append(any_dr_pred)
            torch.cuda.empty_cache()
    all_dr = np.concatenate(all_dr)
    all_dr_pred = np.concatenate(all_dr_pred)
    stats = calculate_metrics(all_dr, all_dr_pred)
    loss = np.mean(all_losses)
    accuracy = np.concatenate(all_predictions).mean()
    return loss, accuracy, stats


def train(args):
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    # model_type = 
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.to(device)
    dataset_size = pd.read_csv("./train.csv").shape[0]
    train_idx, val_idx = train_test_split(
        np.arange(dataset_size), random_state=49)
    try:
        with open("./dataset1.pt", 'rb') as f:
            dataset = torch.load(f)
    except FileNotFoundError:
        dataset = RetinaDataset("./train_images", "./train.csv", "train")
        with open("./dataset1.pt", "wb") as f:
            torch.save(dataset, f)
    train_dataset = RetinaSubsetDataset(
        dataset, subset=train_idx, transform=args.train_transform)
    val_dataset = RetinaSubsetDataset(
        dataset, subset=val_idx, transform=args.val_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size)

    os.makedirs("./train_logs/", exist_ok=True)
    os.makedirs("./val_logs/", exist_ok=True)
    train_writer = SummaryWriter("./train_logs")
    val_writer = SummaryWriter("./val_logs")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True)
    global_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=4, mode='max')
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0
    best_acc = -1
    if args.train:
        for epoch in range(args.epochs):
            print("Validating...")
            val_loss, val_acc, stats = evaluate(
                model, val_dataloader, criterion, device)
            print(json.dumps(stats, indent=4))
            val_writer.add_scalar("acc", val_acc, global_step=global_step)
            val_writer.add_scalar("loss", val_loss, global_step=global_step)
            val_writer.add_scalar("lr", get_lr(optimizer),
                                  global_step=global_step)
            print("Val loss, val acc =", val_loss, val_acc)
            print("Learning rate = ",  get_lr(optimizer))
            if val_acc > best_acc:
                best_acc = val_acc
                print("Saving checkpoint...")
                with open(os.path.join(args.checkpoints_dir, "best.pt"), "wb") as f:
                    torch.save({"model_state_dict": model.state_dict(),
                                "epoch": epoch}, f)
            print(f"Training, epoch {epoch + 1}")
            model.train()
            for image_id, images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device).long()
                optimizer.zero_grad()
                logits = model.forward(images)
                loss = criterion(logits, labels)
                predictions = logits.argmax(dim=1)
                accuracy_t = torch.mean((predictions == labels).float()).item()
                if global_step == 0:
                    train_acc = accuracy_t
                else:
                    train_acc = 0.7 * train_acc + 0.3 * accuracy_t
                loss.backward()
                optimizer.step()
                if global_step % args.log_each == 0:
                    train_writer.add_scalar(
                        "loss", loss.item(), global_step=global_step)
                    train_writer.add_scalar(
                        "acc", train_acc, global_step=global_step)
                global_step += 1
            global_scheduler.step(val_acc)
    else:
        with open(f"{args.checkpoints_dir}/best.pt", "rb") as f:
            model.load_state_dict(torch.load(
                f, map_location=device)["model_state_dict"])
        val_loss, val_acc, stats = evaluate(
            model, val_dataloader, criterion, device)
        print(val_loss, val_acc)
        print(json.dumps(stats, indent=4))


def main(args=None):
    train(args)
    return None


if __name__ == "__main__":
    args = AttrDict({
        "lr": 0.001,
        "batch_size": 8,
        "checkpoint_each": 10,
        "epochs": 100,
        "checkpoints_dir": "./checkpoints_resnet18",
        "log_each": 200,
        "train_transform": Compose([
            Resize((512, 512)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomPerspective(),
            ToTensor(),
        ]),
        "val_transform": Compose([
            Resize((512, 512)),
            ToTensor()
        ]),
        "train": True,
        "model_type": "resnet18"
    })
    main(args)
