import numpy as np
from sklearn import *
import pandas as pd
from pathlib import Path
import random
from PIL import Image
import cv2
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")
import time
import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='output.log', mode='a')
formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

# Download data
def dataset_link():
    pass

zip_path = Path("project_data.zip")
file_path = Path("project_data")

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
trainY = []
train_path = file_path / 'train'

train_fake_image_paths = list(train_path.glob("*/*/*/*.jpg"))
trainY = np.ones_like(train_fake_image_paths)
train_real_image_paths = list(train_path.glob("*/*/*/*/*.jpg"))
trainY = np.append(trainY, np.zeros_like(train_real_image_paths))
train_image_path = train_fake_image_paths + train_real_image_paths
logging.info(len(train_fake_image_paths), len(train_real_image_paths), len(train_image_path), len(trainY))

# validation data
val_fake_image_paths = []
val_real_image_paths = []
valY = []
val_path = file_path / 'val'

val_fake_image_paths = list(val_path.glob("*/*/*/*.jpg"))
valY = np.ones_like(val_fake_image_paths)
val_real_image_paths = list(val_path.glob("*/*/*/*/*.jpg"))
valY = np.append(valY, np.zeros_like(val_real_image_paths))
val_image_path = val_fake_image_paths + val_real_image_paths
logging.info(len(val_fake_image_paths), len(val_real_image_paths), len(val_image_path), len(valY))

# Get train and validation data

def process_image(filename):
    image = cv2.imread(str(filename))
    resized_image = cv2.resize(image, (28, 28))
    permuted_image = np.transpose(resized_image, (2, 0, 1))
    return permuted_image

def get_images(file_paths):
    image_list = []
    for filename in  tqdm(file_paths):
        image_list.append(process_image(filename))
    return image_list

trainX = get_images(train_image_path)
valX = get_images(val_image_path)

trainX = np.array(trainX)
valX = np.array(valX)
print(trainX.shape, valX.shape)

trainXn = trainX.reshape(trainX.shape[0], -1)
valXn = valX.reshape(valX.shape[0], -1)

trainY = trainY.astype('int')
valY = valY.astype('int')

print(trainXn.shape, valXn.shape)

# Model

clfs = {}
trainAE = {}
testAE = {}
testAccuracy = {}
testRecall = {}
testPrecision = {}
testF1 = {}
testROCAUC = {}
testHammingloss = {}
use_time = {}

def timer(func):
    def wrap(model):
        start_time = time.time()
        func(model)
        end_time = time.time()
        total_time = end_time - start_time
        hour = int(total_time//3600)
        min = int((total_time%3600)//60)
        sec = (total_time%60)
        total_time_str = f"{sec:.2f} second{'s' if sec > 1 else ''}" if total_time < 60 else f"{min} minute{'s' if min > 1 else ''} and {sec} second{'s' if sec > 1 else ''}" if total_time < 3600 else f"{hour} hour{'s' if hour > 1 else ''} {min} minute{'s' if min > 1 else ''} {sec} second{'s' if sec > 1 else ''}"
        
        logging.info("Time used for %s: %s", type(model).__name__, total_time_str)
        use_time[type(model).__name__] = total_time_str
    return wrap

def evaluation(model, train_x=trainXn, train_y=trainY, test_x=valXn, test_y=valY,  trainAE=trainAE, testAE=testAE):
    pred_y = model.predict(test_x)
    train_AE = metrics.mean_absolute_error(train_y, model.predict(train_x))
    test_AE  = metrics.mean_absolute_error(test_y, pred_y)
    test_accuracy = metrics.accuracy_score(test_y, pred_y)
    test_recall = metrics.recall_score(test_y, pred_y)
    test_precision = metrics.precision_score(test_y, pred_y)
    test_f1 = metrics.f1_score(test_y, pred_y)
    test_roc_auc = metrics.roc_auc_score(test_y, pred_y)
    test_hamming_loss = metrics.hamming_loss(test_y, pred_y)
    logging.info("{}: train error = {}".format(type(model).__name__, train_AE))
    logging.info("{}: test error = {}".format(type(model).__name__, test_AE))
    logging.info("{}: accuracy = {}".format(type(model).__name__, test_accuracy))
    logging.info("{}: recall = {}".format(type(model).__name__, test_recall))
    logging.info("{}: precision = {}".format(type(model).__name__, test_precision))
    logging.info("{}: F1-score = {}".format(type(model).__name__, test_f1))
    logging.info("{}: ROC and AUC score = {}".format(type(model).__name__, test_roc_auc))
    logging.info("{}: Hamming Loss = {}".format(type(model).__name__, test_hamming_loss))
    trainAE[type(model).__name__] = train_AE
    testAE[type(model).__name__] = test_AE
    testAccuracy[type(model).__name__] = test_accuracy
    testRecall[type(model).__name__] = test_recall
    testPrecision[type(model).__name__] = test_precision
    testF1[type(model).__name__] = test_f1
    testROCAUC[type(model).__name__] = test_roc_auc
    testHammingloss[type(model).__name__] = test_hamming_loss

@timer
def modelling(model):
    match type(model).__name__:
        case "KMeans": paragrid = {'n_clusters': range(2, 10+1)}
        case "KNeighborsClassifier": paragrid = {'n_neighbors': range(1, 10+1)}
        case "KernelRidge": paragrid = {'kernel': ['rbf', 'poly'], 'alpha': np.logspace(-4, 4, 20)}
        case "RandomForestRegressor": paragrid = {'n_estimators': np.logspace(1, 3, 3).astype(int)}
        case "SVR": paragrid = {'C': np.logspace(-2, 2, 5), 'epsilon': np.logspace(-3, 0, 4)}
        case "LinearSVC": paragrid = {'C': [0.1, 1, 10],
                                      'penalty': ['l1', 'l2'],
                                      'max_iter': [1000, 5000, 10000]}
        case "SGDClassifier": paragrid = {'penalty': ['l1', 'l2'],
                                          'alpha': [0.0001, 0.001, 0.01],
                                          'max_iter': [1000, 5000, 10000]}
        case "BaggingClassifier": paragrid = {'base_estimator__criterion': ['gini', 'entropy'], 
                                              'base_estimator__max_depth': [2,4,6,8,10,12],
                                              'n_estimators': list(range(10, 50+1, 10)),
                                              'max_samples': np.arange(0.5, 1, 0.1),
                                              'max_features': np.arange(0.5, 1, 0.1)}
    
    logging.info(f"Now {type(model).__name__}...")
    grid = model_selection.GridSearchCV(model, param_grid=paragrid, cv=5, n_jobs=-1, verbose=2)
    grid.fit(trainXn, trainY)
    clfs[type(model).__name__] = grid.best_estimator_
    clfs[type(model).__name__].fit(trainXn, trainY)
    logging.info(clfs[type(model).__name__])
    evaluation(clfs[type(model).__name__])

models = [cluster.KMeans(), neighbors.KNeighborsClassifier(), kernel_ridge.KernelRidge(), 
          ensemble.RandomForestRegressor(random_state=42), svm.SVR(), svm.LinearSVC(), 
          linear_model.SGDClassifier(), ensemble.BaggingClassifier(tree.DecisionTreeClassifier())]


for model in models:
    modelling(model=model)

# Output results to xlsx

results_dict = {
    'clfs': clfs,
    'trainAE': trainAE,
    'testAE': testAE,
    'Test Accuracy': testAccuracy,
    'Recall Score': testRecall,
    'Precision Score': testPrecision,
    'F1-Score': testF1,
    'ROC & AUC Score': testROCAUC,
    'Hamming Loss': testHammingloss,
    'Time used': use_time
}
results_df = pd.DataFrame.from_dict(results_dict, orient='columns')

logging.info(results_df)
try:
    results_df.to_excel('results.xlsx', index = False)
except:
    import os
    import subprocess
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    env_name
    subprocess.run(["conda", "activate", env_name], shell=True)
    subprocess.run(["pip", "install", "openpyxl"])
    results_df.to_excel('results.xlsx', index = False)
logging.info("\nResults is saved into an xlsx file")