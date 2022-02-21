import requests
import zipfile

import numpy as np
import matplotlib.pyplot as plt

def get_data():
    print('Downloading started')
    url = 'http://bbci.de/competition/download/competition_iv/BCICIV_1calib_1000Hz_mat.zip'

    username = 'replace_with_your_own_username'
    password = 'replace_with_your_own_password'
    req = requests.get(url, auth=(username,password))
    filename = url.split('/')[-1]

    with open(filename,'wb') as output_file:
        output_file.write(req.content)
    print('Downloading Completed')

    # Change to your path
    print('Unzipping')
    path_to_zip_file = 'BCICIV_1calib_1000Hz_mat.zip'
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('data/')
    print('Unzipping Completed')
    return

def parse_log(fp):
    log = {
        'log_info': fp,
        'train_loss': [],
        'val_loss': [],
        'train_acc':[],
        'val_acc': []
    }
    f = open(fp, 'r')
    lines = f.readlines()
    for line in lines:
        if 'train Loss' in line:
            line = line.split()
            log['train_loss'].append(float(line[2]))
            log['train_acc'].append(float(line[4]))
        elif 'val Loss' in line:
            line = line.split()
            log['val_loss'].append(float(line[2]))
            log['val_acc'].append(float(line[4]))
    return log

def plot_log(fp):
    log = parse_log(fp)
    f, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].plot(log['train_acc'], label='Training Accuracy', linestyle='dashed')
    ax[0].plot(log['val_acc'], label='Validation Accuracy', linestyle='dashed')
    ax[0].legend()
    ax[1].plot(log['train_loss'], label='Training Loss', linestyle='dashed')
    ax[1].plot(log['val_loss'], label='Validation Loss', linestyle='dashed')
    ax[1].legend()
    f.suptitle(f'Experiment Log: {fp}')
