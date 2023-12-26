import os
import cv2
import sys
import time
import joblib
import shutil
import random
import logging
import platform
import numpy as np
from sklearn import *
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from itertools import combinations
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, Subset # "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torchvision.datasets import DatasetFolder
import warnings
warnings.filterwarnings("ignore")

def get_data(model=models.resnet18(pretrained=True)):
    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=f'{model._get_name().lower()}.log', mode='a')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Download data
    def dataset_link():
        pass

    zip_path = Path("../project_data.zip")
    file_path = Path("../project_data")

    def download_zip():
        logging.info("Now download data...")
        try:
            import gdown
        except:
            import os
            import subprocess
            env_name = os.environ.get('CONDA_DEFAULT_ENV')
            env_name
            subprocess.run(["conda", "activate", env_name], shell=True)
            subprocess.run(["pip", "install", "gdown"])

            import gdown
        # from Dataset import dataset_link # a py file returning the google drive url which stores data
        url = dataset_link()
        output = zip_path
        gdown.download(url, str(output), quiet=False)
        logging.info("Data downloaded.")

    def unzip_file(zip_path, file_path):
        try:
            from zipfile import ZipFile
        except:
            from zipfile39 import ZipFile
        import shutil
        with ZipFile(str(zip_path), "r") as zip_ref:
            logging.info('Unzipping dataset...')
            zip_ref.extractall(str(file_path))
        logging.info("Data unzipped.")
        if Path('__MACOSX').exists():
            shutil.rmtree('__MACOSX')

    if Path(file_path).is_dir():
        logging.info(f"{file_path} exists.")
    elif Path(zip_path).exists():
        unzip_file(zip_path, file_path)
    else:
        download_zip()
        unzip_file(zip_path, file_path)

    # Extract Data

    # train data
    train_fake_image_paths = []
    train_real_image_paths = []
    train_real_image = []
    train_image_path = []
    trainY = []
    train_path = file_path / 'train'

    train_real_image_paths = list(train_path.glob("*/*/*/*/*.jpg"))
    for i in range(5):
        train_fake_image_paths.append(list(train_path.glob(f"FakeManipulation-{i+1}/*/*/*.jpg")))
        trainY.append(np.ones_like(train_fake_image_paths[i]))
        # random.seed(42)
        train_real_image.append(random.choices(train_real_image_paths, k=len(train_fake_image_paths[i])))
        trainY[i] = np.append(trainY[i], np.zeros_like(train_real_image[i]))
        train_image_path.append(train_fake_image_paths[i] + train_real_image[i])

    for i in range(5):
        print("[Train] FakeManipulation-{}".format(i+1))
        print("No. of train fake images:", len(train_fake_image_paths[i]), 
            "\nNo. of train real images", len(train_real_image[i]), 
            "\nTotal train images:", len(train_image_path[i]), 
            "\nTotal train labels", len(trainY[i]), end='\n')
        print(f"{'='*35}")
        
    # validation data
    val_fake_image_paths = []
    val_real_image_paths = []
    val_real_image = []
    valY = []
    val_image_path = []
    val_path = file_path / 'val'
    val_real_image_paths = list(val_path.glob("*/*/*/*/*.jpg"))

    for i in range(5):
        val_fake_image_paths.append(list(val_path.glob(f"FakeManipulation-{i+1}/*/*/*.jpg")))
        valY.append(np.ones_like(val_fake_image_paths[i]))
        # random.seed(42)
        val_real_image.append(random.choices(val_real_image_paths, k=len(val_fake_image_paths[i])))
        valY[i] = np.append(valY[i], np.zeros_like(val_real_image[i]))
        val_image_path.append(val_fake_image_paths[i] + val_real_image[i])

    for i in range(5):
        print("[Val] FakeManipulation-{}".format(i+1))
        print("No. of validation fake images:", len(val_fake_image_paths[i]), 
        "\nNo. of validation real images", len(val_real_image[i]), 
        "\nTotal validation images:", len(val_image_path[i]), 
        "\nTotal validation labels", len(valY[i]), end='\n')
        print(f"{'='*35}")

    return train_image_path, trainY, val_image_path, valY

def modelling(model=models.resnet18(pretrained=True), i_comb=(0,), train_image_path=[], train_Y=[], val_image_path=[], val_Y=[]):
    logging.info("Now test on Fake-Manipulation-{}".format(i_comb))

    original_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((28, 28)),  # Resize the image to (28, 28)
        # transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ColorJitter(contrast=2.0),  # Enhance contrast
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    mirror_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((28, 28)),  # Resize the image to (28, 28)
        # transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(p=1.0),  # Mirror the image horizontally
        transforms.ColorJitter(contrast=2.0),  # Enhance contrast
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    trainX = []
    valX = []
    train_path = []
    trainY = []
    val_path = []
    valY = []
    

    for i in i_comb:
        train_path += train_image_path[i]
        trainY += list(train_Y[i])
        val_path += val_image_path[i]
        valY += list(val_Y[i])

    def process_image_and_duplicate_label(filename, label, mode=0):
        image = cv2.imread(str(filename))
        resized_image = original_transform(image) if mode==0 else mirror_transform(image)
        return resized_image, label

    trainX, trainY = zip(*joblib.Parallel(n_jobs=-1)(joblib.delayed(process_image_and_duplicate_label)(filename, label, mode) for mode in [0, 1] for filename, label in zip(tqdm(train_path), trainY)))
    valX, valY = zip(*joblib.Parallel(n_jobs=-1)(joblib.delayed(process_image_and_duplicate_label)(filename, label, mode) for mode in [0, 1] for filename, label in zip(tqdm(val_path), valY)))

    trainX = torch.stack(trainX)  # Convert the list of tensors to a tensor
    valX = torch.stack(valX)  # Convert the list of tensors to a tensor

    logging.info("TrainX shape: %s, ValX shape: %s", trainX.shape, valX.shape)

    trainY = np.array(trainY, dtype=np.int64)
    trainY = torch.tensor(trainY, dtype=torch.long)

    valY = np.array(valY, dtype=np.int64)
    valY = torch.tensor(valY, dtype=torch.long)

    train_dataset = TensorDataset(trainX, trainY)

    # Model
    model_name = model._get_name()
    logging.info(model_name)
    if model_name.lower() == 'resnet':
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
    elif model_name.lower() == 'efficientnet':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    for param in model.parameters():
        param.requires_grad = True

    torch.cuda.empty_cache()
    # "cuda" only when GPUs are available.
    device = torch.device("mps" if platform.system() == 'Darwin' and torch.backends.mps.is_built() \
                else "cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info("device: %s", device)
    
    # Initialize a model, and put it on the device specified.
    model = model.to(device)
    model.device = device

    # Convert the data to the appropriate device
    trainX = trainX.to(device)
    trainY = trainY.to(device)
    valX = valX.to(device)
    valY = valY.to(device)

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()
    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # The number of training epochs/ training loop
    n_epochs = 20
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_record = {'train': [], 'dev': []}
    acc_record = {'train': [], 'dev': []}

    accumulation_steps = 4  # Number of batches to accumulate gradients over

    # best_loss = float('Inf')
    # best_val_loss = float('inf')
    best_accuracy = -float('inf')
    best_val_accuracy = -float('inf')
    patience = 10  # Number of epochs to wait before stopping
    epochs_no_improve = 0

    for epoch in tqdm(range(n_epochs)):
        running_loss = 0.0

        model.train() # Make sure the model is in train mode before training.
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
        
        loss_record['train'].append(running_loss / len(train_loader))

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_outputs = model(valX)
            val_loss = criterion(val_outputs, valY)
            val_accuracy = (val_outputs.argmax(dim=1) == valY).float().mean()
            acc_record['train'].append(val_accuracy.cpu().numpy())

        logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy.item():.4f}")
        
        # # Early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == patience:
        #         logging.info("Early stopping!")
        #         break  # Early stop

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # if epochs_no_improve == patience:
                # logging.info("Early stopping!")
                # break  # Early stop


    logging.info("Training finished.")
    if best_accuracy > best_val_accuracy:
        best_accuracy = best_val_accuracy
        torch.save(model.state_dict(), "best_model.pt")

    import matplotlib.pyplot as plt

    # Initialize figures for loss and accuracy
    fig_loss, ax_loss = plt.subplots()
    fig_acc, ax_acc = plt.subplots()

    # Update the learning curve plots
    ax_loss.plot(loss_record['train'], label='Training Loss')
    ax_loss.plot(loss_record['dev'], label='Validation Loss')

    # Update the learning curve plots
    ax_loss.plot(loss_record['train'], label='Training Loss')
    ax_acc.plot(acc_record['train'], label='Training Accuracy')

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Learning Curve - Loss')
    ax_loss.legend()

    ax_acc.plot(acc_record['train'], label='Training Accuracy')
    ax_acc.plot(acc_record['dev'], label='Validation Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Learning Curve - Accuracy')
    ax_acc.legend()

    # Pause the plot to update
    plt.pause(0.01)

    plt.show()
    plt.savefig(f"Combination: {i_comb}.png")

def main():
    train_image_path, trainY, val_image_path, valY = get_data()
    print(type(train_image_path))

if __name__ == '__main__':
    main()