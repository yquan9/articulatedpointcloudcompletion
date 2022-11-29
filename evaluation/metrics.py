from torch_cluster import knn
from torch import set_grad_enabled
from torch.nn.functional import mse_loss
from trimesh import Trimesh

def sample_wise_cd(x,y):
    set_grad_enabled(False)
    tsrT1 = mse_loss(y[knn(y,x,1)[1]], x, reduction="sum").item()
    tsrT2 = mse_loss(x[knn(x,y,1)[1]], y, reduction="sum").item()
    tplRst = (tsrT1,tsrT2,tsrT1+tsrT2)
    return tplRst

def sample_wise_volerr(x,y,fce):
    scaTSmpVol = Trimesh(vertices=y.cpu().detach().numpy(),faces=fce).volume
    return abs(Trimesh(vertices=x.cpu().detach().numpy(),faces=fce).volume-scaTSmpVol) / scaTSmpVol
