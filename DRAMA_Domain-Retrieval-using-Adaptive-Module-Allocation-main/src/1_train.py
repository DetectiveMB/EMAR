import logging
import os

import hydra
import numpy as np

from omegaconf import DictConfig
from torch import autocast, no_grad
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import MultiDomainDataset
from model.utils import seed_everything
from model.model import BiEncoder, CrossEncoder
from model.loss import TripletMarginLoss, CrossEncoderLoss


logger = logging.getLogger(__name__)


def train(
        train_dataloader, 
        model, 
        optimizer, 
        loss_fn, 
        device,
        epoch
    ) -> float:

    losses = []
    accuracies = []

    optimizer.zero_grad()

    pbar = tqdm(train_dataloader)
    autocast_device = 'cuda' if 'cuda' in device else 'cpu'
    for batch in pbar:
        with autocast(device_type=autocast_device):
            output = model(batch)
        if 't5' in model.model_name:
            loss_val = output.loss
            accuracy = output['accuracy']
        else:
            loss_val, accuracy = loss_fn(output)

        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        accuracies.extend(accuracy.tolist())
        
        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(accuracies)

        summary = "TRAIN EPOCH {:3d} Sim Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_sim_accuracy*100,2), average_loss)
        pbar.set_description(summary)
    logging.info(summary)
    
    return average_loss

def validate(
        train_dataloader, 
        model,  
        loss_fn, 
        device,
        epoch
    ) -> float:

    losses = []
    accuracies = []

    autocast_device = 'cuda' if 'cuda' in device else 'cpu'
    pbar = tqdm(train_dataloader, colour="red")
    for batch in pbar:
        with autocast(device_type=autocast_device):
            with no_grad():
                output = model(batch)
        if 't5' in model.model_name:
            loss_val = output.loss
            accuracy = output['accuracy']
        else:
            loss_val, accuracy = loss_fn(output)

        losses.append(loss_val.cpu().detach().item())
        accuracies.extend(accuracy.tolist())
        
        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(accuracies)

        summary = "VAL EPOCH {:5d} Sim Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_sim_accuracy*100,2), average_loss)
        pbar.set_description(summary)
    logging.info(summary)
    
    return average_loss


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_dir = cfg.dataset.data_dir
    
    logging_dir = cfg.logging.logging_dir
    os.makedirs(logging_dir, exist_ok=True)
    model_dir = cfg.dataset.model_dir
    os.makedirs(model_dir, exist_ok=True)
    logging_file = os.path.join(logging_dir, '1_train.log')
    logging.basicConfig(filename=logging_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )
    
    seed_everything(cfg.general.seed)

    train_dataset = MultiDomainDataset(data_dir, split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle
    )
    
    val_dataset = MultiDomainDataset(data_dir, split='val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle
    )
    if 'cross' in cfg.model.model_name:
        model = CrossEncoder(
            cfg.model.model_name,
            cfg.model.tokenizer_name,
            cfg.model.max_tokens,
            cfg.model.device
        )
    else:
        model = BiEncoder(
            cfg.model.model_name,
            cfg.model.tokenizer_name,
            cfg.model.max_tokens,
            cfg.model.normalize,
            cfg.model.pooling_mode,
            cfg.model.device
        )
        # model.load(os.path.join(model_dir,f'{cfg.model.name}_epoch_20.pt'))
    model.save(os.path.join(model_dir,f'{cfg.model.name}.pt'))
    if 'cross' in cfg.model.model_name:
        loss_fn = CrossEncoderLoss()
    else:
        loss_fn = TripletMarginLoss(cfg.dataset.loss_margin) # MultipleRankingLoss(device=cfg.model.device)
    optimizer = AdamW(model.parameters(), lr=cfg.dataset.lr)
    best_val_loss = validate(val_dataloader, model, loss_fn, cfg.model.device, -1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)
    bad_epochs = 0
    for epoch in tqdm(range(cfg.dataset.max_epoch)):
        model.train()
        train_loss = train(train_dataloader, model, optimizer, loss_fn, cfg.model.device, epoch + 1)
        model.eval()
        val_loss = validate(val_dataloader, model, loss_fn, cfg.model.device, epoch + 1)
        # scheduler.step(val_loss)

        if val_loss < best_val_loss:
            logging.info(f'Found new best model on epoch: {epoch + 1}, new best validation loss {val_loss}')
            best_val_loss = val_loss
            logging.info(f'saving model checkpoint at epoch {epoch + 1}')
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs == cfg.general.patience:
                logging.info(f'Model did not improve for {bad_epochs} epochs. Stopping the training at EPOCH: {epoch + 1}')
                exit()
        # if 't5' in cfg.model.model_name:
        #     model.save_adapter(os.path.join(model_dir, f'{cfg.dataset.name}'), f'{cfg.dataset.name}_adapter')
        # else:
        model.save(os.path.join(model_dir,f'{cfg.model.name}.pt'))
        

    
if __name__ == '__main__':
    main()
