import csv
import dill
import torch
import wandb
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from util import * 
from losses import EnergyLoss, PosForceLoss
from scripts.commom_util import *

def get_configs(model_path):
    configs = dict()
    with open(f"{model_path}/info.txt") as fp:
        for line in fp:
            line = line.strip().split(" ")
            if len(line)>1:
                configs[line[0]] = line[1]

    return configs

def predict(args, configs, device):
    # initialize
    dataset = get_dataset(configs["dataset"], {"style":configs["style"], "molecule":configs["molecule"], "split":configs["split"]}, "test", args.root)
    identifier = dataset.identifier
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate)
    
    path = glob.glob(f"{args.model_path}/{args.epoch:03d}_*.pth")
    if not path: raise ValueError("model not found")
    model = torch.load(path[0], map_location=torch.device(torch.cuda.current_device()), pickle_module=dill)
    print(f"Using model {path[0]}")
    model.eval()
    
    # test
    preds = []
    losses = []
    tq = tqdm(test_dataloader)
    for data, label in tq:
        data = {i:v.to(device) for i, v in data.items()}
        label = {i:v.to(device) for i, v in label.items()}
        pred = model(data)
        preds.append( ((torch.cat((pred["E"], pred["F"].reshape(1, -1)), axis=1)).squeeze()).tolist() )
     
        loss_E = (EnergyLoss(pred, label).to("cpu").item())
        loss_F = PosForceLoss(pred, label)
        loss_F = [ (l.to("cpu").item()) for l in loss_F]
        losses.append([loss_E]+loss_F)
    
    # save result
    save_reult(args.model_path, identifier, preds, losses)

def metric(args, configs):
    # Every dataset has different unit, unify them to Energy:eV and Force: eV/A
    toEv = {"MD17SingleDataset":0.0433634, "MD17Dataset":0.0433634}
    scale = toEv[configs["dataset"]]

    # load and calculate
    style, molecule, split = configs["style"], configs["molecule"], configs["split"]
    with open(f"{args.model_path}/loss_MD17SingleDataset_{style}_{molecule}{split}test.csv", newline='') as fp:
        cdata = list(csv.reader(fp, quoting=csv.QUOTE_NONNUMERIC))
        cdata = [[c for c in row] for row in cdata]
        loss_Emole , loss_Fmole= [], []
        hit, total = 0, len(cdata)
        failE, failF = 0, 0

        for row in tqdm(cdata):
            loss_Emole.append(row[0]) 
            loss_Fmole.append(row[1:])
            m = max(row[1:])
            if row[0]<=0.02/scale and m<=0.03/scale: hit += 1
            else:
                if row[0]>0.02/scale: failE += 1
                if m>0.03/scale: failF += 1
        loss_Emole , loss_Fmole= np.array(loss_Emole), np.array(loss_Fmole)

        print(f"Energy MAE: {np.mean(loss_Emole):.3f},\tForce MAE: {np.mean(loss_Fmole):.3f},\tEFwT: {hit/total:.3f} (failE:{(failE/total):.3f}, failF:{(failF/total):.3f})\n")    
        if args.wandb:
            wandb.run.summary["Emae"] = f"{np.mean(loss_Emole):.3f}"
            wandb.run.summary["Fmae"] = f"{np.mean(loss_Fmole):.3f}"
            wandb.run.summary["EFwT"] = f"{hit/total:.3f}"

def main(args):
    # set device
    if torch.cuda.is_available():
        print(f"GPU num detected: {torch.cuda.device_count()}")
        print(f"Using GPU {torch.cuda.current_device()}")
        device = "cuda"
    else:
        print(f"Using CPU")
        device = "cpu"

    # script flow contorl
    try:
        configs = get_configs(args.model_path)
        predict(args, configs, device)
        metric(args, configs)

    # error handle
    except BaseException as e:
        print(e)
        traceback.print_exc()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # env
    parser.add_argument("--root", type=str, default="../../datasets/MD17/datas", help="path to the root dir of dataset")
    parser.add_argument("--model_path", type=str, help="path to the trained model dir")

    # config
    parser.add_argument("--epoch", type=int, default=299, help="checkpoint of which epoch to test")
    
    main(parser.parse_args())