import torch
from torch.utils.tensorboard import SummaryWriter

def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)