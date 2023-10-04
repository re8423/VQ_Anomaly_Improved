from pathlib import Path
from typing import Any, Dict

import numpy as np
from IPython.display import display
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from torchvision.transforms import ToPILImage

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks

from anomalib.data.folder import Folder, FolderDataset
from anomalib.pre_processing import PreProcessor
pre_process = PreProcessor(image_size=256, to_tensor=True)

import torch

#Script to run DFKDE model from anomalib

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium' or 'high')
    MODEL = "dfkde"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
    CONFIG_PATH = f"/home2/mvdk66/Anomaly_main/anomalib/anomalib/models/{MODEL}/config.yaml"
    with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
        print(file.read())
        
    # pass the config file to model, callbacks and datamodule
    config = get_configurable_parameters(config_path=CONFIG_PATH)
    # config["dataset"]["path"] = "../../datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/New/" 
    config["dataset"]["path"] = "/home2/mvdk66/Anomaly_main" 
    folder_datamodule = Folder(
        root="/home2/mvdk66/Anomaly_main/",
        normal_dir="./UCSD_Train",
        abnormal_dir="./UCSD_Test",
        task="classification",
        image_size=128,
        num_workers=4,
        train_batch_size=10,
        eval_batch_size=10,
        seed=1,
    )

    folder_datamodule.setup()
    folder_datamodule.prepare_data()
    
    model = get_model(config)
    callbacks = get_callbacks(config)
    
    callbacks.append(RichProgressBar())
    
    # start training
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=folder_datamodule)
    
#     s_dict = torch.load("/home2/mvdk66/Anomaly_main/anomalib/notebooks/Prev/results/ganomaly/UCSD/run/weights/ga_cp_2")
#     del s_dict['image_metrics.AnomalyScoreThreshold.value']
#     model.load_state_dict(s_dict)
    torch.save(model.state_dict(), '/home2/mvdk66/Anomaly_main/anomalib/notebooks/Prev/results/dfkde/UCSD/run/weights/dfkde1')
    print('model saved!')
    
#     load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
#     trainer.callbacks.insert(0, load_model_callback)
    trainer.test(model=model, datamodule=folder_datamodule)
    
    
    