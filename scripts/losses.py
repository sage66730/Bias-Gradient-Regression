import torch

# after squeeze:
# prediction of force: [nA*batch_size, 3]
# label of force: [batch_size, nA, 3]

# Total Loss (E+F) ########################################################################################################
def Fz_FELoss(pred, label, train_stat):
    if train_stat["cur_epoch"] < train_stat["freeze_epoch"]:
        return AtomForceLoss(pred, label)
    else:   
        return EnergyLoss(pred, label)

def F_ELoss(pred, label, train_stat):
    if train_stat["cur_epoch"] < 150:
        return AtomForceLoss(pred, label)
    else:   
        return EnergyLoss(pred, label)

def F50_E10Loss(pred, label, train_stat):
    if train_stat["cur_epoch"]%60 < 50:
        return AtomForceLoss(pred, label)
    else:   
        return EnergyLoss(pred, label)

def F10_E10Loss(pred, label, train_stat):
    if train_stat["cur_epoch"]%20 < 10:
        return AtomForceLoss(pred, label)
    else:   
        return EnergyLoss(pred, label)

def E_FLoss(pred, label, train_stat):
    if train_stat["cur_epoch"] < 150:
        return EnergyLoss(pred, label)
    else:   
        return AtomForceLoss(pred, label)
        
def EnergyForceLoss(pred, label, train_stat):
    alpha = train_stat["alpha"]
    E = EnergyLoss(pred, label)
    F = AtomForceLoss(pred, label)
    return E + alpha*F

# Enery Loss ########################################################################################################
def EnergyLoss(pred, label):
    p, l = pred["E"].squeeze(), label["E"].squeeze()
    mae = torch.nn.L1Loss()
    return mae(p, l)

# Force Loss ########################################################################################################
def PosForceLoss(pred, label):
    # not for training, only for predicting to calculate EFwT
    p, l = pred["F"], label["F"]
    batch_size = l.shape[0]   
    p = p.reshape(batch_size, -1).t()
    l = l.reshape(batch_size, -1).t()

    mae = torch.nn.L1Loss()
    loss = [mae(p[i], l[i]) for i in range(len(p))]
    return loss

def AtomForceLoss(pred, label):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 3))
    l = l.reshape((-1, 3))
    mae = torch.nn.L1Loss()
    f = mae(p, l)
    return f

if __name__ == "__main__":
    p = {"F":torch.rand((25,3))}
    l = {"F":torch.rand((5,5,3))}

    PF = PosForceLoss
    AF = AtomForceLoss

    lossPF = PF(p, l)
    lossAF = AF(p, l)

    print(lossPF)
    print(sum(lossPF)/len(lossPF))
    print(lossAF)
