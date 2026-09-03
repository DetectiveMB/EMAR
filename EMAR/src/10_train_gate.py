import logging
import os

import hydra
import numpy as np

from adapters import BnConfig
from omegaconf import DictConfig
from torch import no_grad
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import GateDomainDataset
from model.utils import seed_everything
from model.gated_temp import GatedCLS
from model.loss import MoEGateCrossEntropyLoss


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
    gate_accuracies = []

    optimizer.zero_grad()

    pbar = tqdm(train_dataloader)
    for batch in pbar:
        output = model(batch)
        loss_val, gate_accuracy = loss_fn(output, batch)
            

        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        gate_accuracies.extend(gate_accuracy.tolist())
        
        average_loss = np.mean(losses)
        average_gate_accuracy = np.mean(gate_accuracies)

        summary = "TRAIN EPOCH {:3d}, Gate Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_gate_accuracy*100,2), average_loss)
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
    gate_accuracies = []


    pbar = tqdm(train_dataloader, colour="red")
    for batch in pbar:
        with no_grad():
            output = model(batch)
            loss_val, gate_accuracy = loss_fn(output, batch)
        
        losses.append(loss_val.cpu().detach().item())
        gate_accuracies.extend(gate_accuracy.tolist())
        
        average_loss = np.mean(losses)
        average_gate_accuracy = np.mean(gate_accuracies)

        summary = "VAL EPOCH {:3d}, Gate Accuracy {}, Average Loss {:.2e}".format(epoch, round(average_gate_accuracy*100,2), average_loss)
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
    logging_file = os.path.join(logging_dir, '9_train_gate.log')
    logging.basicConfig(filename=logging_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )
    
    seed_everything(cfg.general.seed)
    
    train_dataset = GateDomainDataset(
        data_dir=data_dir,
        label_file=cfg.general.label_file,
        split='train'
    )
    val_dataset = GateDomainDataset(
        data_dir=data_dir,
        label_file=cfg.general.label_file,
        split='val'
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle
    )

    model = GatedCLS(
        cfg.model.model_name,
        cfg.model.tokenizer_name,
        cfg.general.num_classes,
        cfg.model.max_tokens,
        cfg.model.normalize,
        cfg.model.pooling_mode,
        cfg.model.device
    )
    model.load(os.path.join(cfg.general.all_model_dir,f'{cfg.model.name}.pt'))
    
    config = BnConfig(
        mh_adapter=True, 
        output_adapter=True, 
        reduction_factor=cfg.general.reduction_factor, 
        non_linearity="relu"
    )
    model.add_adapter('cls', config)
    
    loss_fn = MoEGateCrossEntropyLoss(num_classes=cfg.general.num_classes)
    optimizer = AdamW(model.parameters(), lr=cfg.dataset.lr)
    best_val_loss = 999

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle
    )
    model.save_adapter(os.path.join(model_dir, f'{cfg.model.name}_cls_adapter'), 'cls')
    model.save_cls(os.path.join(model_dir, f'{cfg.model.name}_cls_head.pt'))

    for epoch in tqdm(range(cfg.dataset.max_epoch)):
        model.train()
        train_loss = train(train_dataloader, model, optimizer, loss_fn, cfg.model.device, epoch + 1)
        model.eval()
        val_loss = validate(val_dataloader, model, loss_fn, cfg.model.device, epoch + 1)

        if val_loss < best_val_loss:
            logging.info(f'Found new best model on epoch: {epoch + 1}, new best validation loss {val_loss}')
            best_val_loss = val_loss
            logging.info(f'saving model checkpoint at epoch {epoch + 1}')
            model.save_adapter(os.path.join(model_dir, f'{cfg.model.name}_cls_adapter'), 'cls')
            model.save_cls(os.path.join(model_dir, f'{cfg.model.name}_cls_head.pt'))

    
if __name__ == '__main__':
    main()
