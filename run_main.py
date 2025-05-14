import json
import torch
import random
import numpy as np
import torch.nn as nn
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import sys
sys.path.insert(1, os.path.abspath(".."))
codes_dir = os.path.join('../MICCAI2025/', 'src') 

from src.model import GRUModel
from src.data_loader import load_data
from src.utils import CustomEncoder, model_eval

config = {
    "seed": 42,
    "num_epochs": 10,
    "batch_size": 32,
    "model_params": {
        "input_size": 128,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 1,
        "input_dim": 128,
        "embed_dim": 64,
        "nhead": 4,
        "d_hid": 256,
        "nlayers": 2,
        "lr": 0.001
    },
    "data_folder": " ",
    "path_to_patches": ".../__allData/miccai2025/anatomical_patches_16_img224",
    "output_folder": "./output/",
    "normalization": True,
    "seq_length": 97,
    "shuffle": False
}

if __name__ == "__main__":
    torch.manual_seed(config["seed"])
    pl.seed_everything(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    initial_mem = torch.cuda.memory_allocated()
    print(f'Initial memory usage: {initial_mem} bytes')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUModel(**config["model_params"])
    model = model.to(device)

    train_loader, val_loader, test_loader = load_data(data_folder=config['data_folder'],
                                                        path_to_patches=config['path_to_patches'],
                                                        batch_size=config['batch_size'],
                                                        normalization=config['normalization'],
                                                        sequence_length=config["seq_length"],
                                                        shuffle=config["shuffle"])

    SAVE_MODEL_NAME = model.__class__.__name__
    SAVE_MODEL_PATH = config["output_folder"] + SAVE_MODEL_NAME + '_' + datetime.now().strftime("%Y-%m-%d___%H-%M-%S")
    if not os.path.exists(SAVE_MODEL_PATH):
        os.mkdir(SAVE_MODEL_PATH)

    print("DEVICE:", device)
    print('SAVE MODEL PATH:', SAVE_MODEL_PATH)
    print(json.dumps(config, indent=4, cls=CustomEncoder))

    checkpoint_callback = ModelCheckpoint(
                        monitor='val_auc',
                        dirpath= SAVE_MODEL_PATH,
                        filename='best-model',
                        save_top_k=1,
                        mode='max',
                        save_last=True
                    )


    logger = TensorBoardLogger(SAVE_MODEL_PATH, name='name')
    trainer = Trainer(max_epochs=config["num_epochs"],
                    logger=logger, 
                    enable_progress_bar=False,
                    accelerator=device.type,
                    callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print('last model:')
    weights = torch.load(os.path.join(SAVE_MODEL_PATH, 'last.ckpt'), map_location=device)
    model_eval(model, weights, test_loader, device)

    print('best model:')
    weights = torch.load(os.path.join(SAVE_MODEL_PATH, 'best-model.ckpt'), map_location=device)
    model_eval(model, weights, test_loader, device)