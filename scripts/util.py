import csv
import torch

# import loss_fn
from losses import *

# import models
from models.schnet.schnet import SchNetWrap
from models.cgcnn.cgcnn import CGCNN
from models.dimenet2.dimenet2 import DimeNetPlusPlusWrap
from models.gemnet.gemnet import GemNetT
from models.gemnetOC.gemnetOC import GemNetOC
from models.forcenet.forcenet import ForceNet

# import datasets
from datasets.MD17.MD17Dataset import MD17Dataset, MD17SingleDataset

def get_model(model, bias_layer):
    if model == "schnet": return SchNetWrap(regress_forces=True, bias_layer=bias_layer)
    if model == "cgcnn": return CGCNN(regress_forces=True, bias_layer=bias_layer)
    if model == "dimenet2": return DimeNetPlusPlusWrap(regress_forces=True, bias_layer=bias_layer)
    if model == "gemnet": return GemNetT(regress_forces=True, bias_layer=bias_layer)
    if model == "gemnetOC": return GemNetOC(regress_forces=True, bias_layer=bias_layer)
    if model == "forcenet": return ForceNet(regress_forces=True, bias_layer=bias_layer)

    raise ValueError("model name incorrect")

def get_dataset(dataset, dataset_stat, task, root):
    if dataset == "MD17SingleDataset": return MD17SingleDataset(dataset_stat["style"], dataset_stat["molecule"], task, dataset_stat["split"], root)
    if dataset == "MD17Dataset": return MD17Dataset(dataset_stat["style"], dataset_stat["molecule"], task, dataset_stat["split"], root)

    raise ValueError("dataset name incorrect")

def get_loss_fn(loss_fn):
    # baseline losses
    if loss_fn == "EnergyForceLoss": return EnergyForceLoss
    if loss_fn == "EnergyLoss": return EnergyLoss
    if loss_fn == "AtomForceLoss": return AtomForceLoss

    # toggling force and energy loss
    if loss_fn == "F_ELoss": return F_ELoss
    if loss_fn == "F50_E10Loss": return F50_E10Loss
    if loss_fn == "F10_E10Loss": return F10_E10Loss
    if loss_fn == "E_FLoss": return E_FLoss

    raise ValueError("loss function name incorrect")

def get_optimizer(model, optimizer, lr, wd):
    if optimizer == "Adam": return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if optimizer == "SGD": return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    raise ValueError("optimizer name incorrect")

def save_reult(model_path, identifier, preds, losses):
    # save after prediction
    if preds:
        csvfile = open(f"{model_path}/pred_{identifier}.csv", "w", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(preds)

    if losses:
        csvfile = open(f"{model_path}/loss_{identifier}.csv", "w", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(losses)
    return




