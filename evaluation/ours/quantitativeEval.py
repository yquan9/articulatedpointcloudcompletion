from torch_cluster import knn, fps
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from torch import max, device, zeros, bool, reshape, mean, sigmoid, cat, sqrt, save as ptsave
from torch import set_grad_enabled, float, tensor, randint, rand_like, rand, randn, sum, load as ptload
from pickle import load as pkload
from torch.nn.modules import Module
from torch.nn import Linear
from torch.nn.functional import mse_loss, leaky_relu
from torch.optim import Adam
from tqdm import tqdm
from numpy import savetxt, load as npload
from argparse import ArgumentParser
from spiralae.model import SpiralDecoder
import sys
sys.path.append("..")
from metrics import *




class PtlFulShpPair(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PtlFulShpPair, self).__init__(root, transform, pre_transform)
        self.tsrFulMsh = ptload(root+"fulMsh.pth") # read full shapes
        self.tsrMask = ptload(root+"mask.pth") # read masks
        self.tsrMskMap = tensor([9,1,3,5,6],dtype=int)
        self.tsrJntLst = ptload(root+"jointPointPairs.pth").long() # read joint point indices
        self.tsrBinDct = zeros([256,8],dtype=bool) # `dec2bin` mapping
        strBinCod = None
        scaI = 0
        scaJ = 0
        # construct `dec2bin` mapping from 0 to 255 for 8-bit decompression of masks
        while scaI<256:
            strBinCod = bin(scaI)[2:]
            scaJ = 0
            while scaJ<len(strBinCod):
                if strBinCod[scaJ] == "1":
                    self.tsrBinDct[scaI,(8-len(strBinCod))+scaJ] = True
                scaJ += 1
            scaI += 1

    def len(self):
        return self.tsrFulMsh.size(0) * 5

    def get(self,idx):
        tsrTthShp = self.tsrFulMsh[idx//5] # target ground truth
        # get the decimal mask code;
        # convert them to binary code, reshape and take the front 3889 bits as mask code;
        # select points from the targeted ground truth shape according to the mask
        tsrPtlShp = self.tsrFulMsh[idx//5,reshape(self.tsrBinDct[self.tsrMask[idx//5*10+self.tsrMskMap[idx%5]].long(),:],[487*8])[0:3889]]
        tsrJntPnt = mean(reshape(tsrTthShp[self.tsrJntLst],[2,17,3]),0)[[0,1,6,7,8,13,14,15]] # calculate joint point sets
        tsrPtOfSt = mean(tsrPtlShp,0,True)
        tsrPtlShp = tsrPtlShp - tsrPtOfSt # centralize
        tsrJntPnt = tsrJntPnt - tsrPtOfSt # shift according to the partial shape
        tsrTthShp = tsrTthShp - mean(tsrTthShp,0,True) # centralize
        tsrPtlShp = tsrPtlShp + 0.00 * rand_like(tsrPtlShp) # add noise
        return Data(p=tsrPtlShp,s=tsrJntPnt,r=tsrTthShp,num_nodes=tsrPtlShp.size(0))



class PartialShapeEncoder(Module):
    def __init__(self,k=200,dilation=1):
        super(PartialShapeEncoder,self).__init__()
        self.scaK = k
        self.lstNbr = list(range(0,k,dilation))
        self.mlp1 = Linear(3,128)
        self.mlp2 = Linear(128,256)
        self.mlp3 = Linear(512,512)
        self.mlp4 = Linear(512,1024)
        self.mlp5 = Linear(1024,512)
        self.mlp6 = Linear(512,128)
        self.mlp7 = Linear(128,24)

        self.mlp8 = Linear(len(self.lstNbr)*3,256)
        self.mlp9 = Linear(256,256)
        self.mlp10 = Linear(256,64)

        self.mlp11 = Linear(64*8,512)
        self.mlp12 = Linear(512,128)


    def forward(self,x,bchx,bs):
        tsrBchSkl = None
        tsrRst1 = None
        tsrKnnIdx = None
        tplRst = None
        tsrFpsIdx = fps(x,bchx,random_start=False)
        tsrX = leaky_relu(self.mlp1(x[tsrFpsIdx]))
        tsrX = leaky_relu(self.mlp2(tsrX))
        tsrX = cat((global_max_pool(tsrX,bchx[tsrFpsIdx],bs)[bchx[tsrFpsIdx]],tsrX),dim=1)
        tsrX = leaky_relu(self.mlp3(tsrX))
        tsrX = leaky_relu(self.mlp4(tsrX))
        tsrX = global_max_pool(tsrX,bchx[tsrFpsIdx],bs)
        tsrRst1 = reshape(self.mlp7(leaky_relu(self.mlp6(leaky_relu(self.mlp5(tsrX))))),[bs*8,3])

        tsrBchSkl = reshape(reshape(tensor(range(bs),dtype=bchx.dtype),[bs,1]).repeat([1,8]),[bs*8])
        tsrKnnIdx = knn(x[tsrFpsIdx].cpu(),tsrRst1.cpu(),self.scaK,bchx[tsrFpsIdx].cpu(),tsrBchSkl)[1]
        tsrKnnIdx = reshape(tsrKnnIdx,[bs*8,self.scaK])
        tsrKnnIdx = reshape(tsrKnnIdx[:,self.lstNbr],[bs*8*len(self.lstNbr)])
        '''
        savetxt("temp/x.xyz",x.cpu().detach().numpy())
        savetxt("temp/s.xyz",tsrRst1.cpu().detach().numpy())
        savetxt("temp/p.xyz",x[tsrFpsIdx][tsrKnnIdx].cpu().detach().numpy())
        exit()
        '''
        tsrX = leaky_relu(self.mlp8(reshape(x[tsrFpsIdx][tsrKnnIdx],[bs*8,len(self.lstNbr)*3])))
        tsrX = leaky_relu(self.mlp9(tsrX))
        tsrX = leaky_relu(self.mlp10(tsrX))
        tsrX = reshape(tsrX,[bs,8*64])
        tsrX = leaky_relu(self.mlp11(tsrX))
        tsrX = self.mlp12(tsrX)

        tplRst = (tsrRst1,tsrX)
        return tplRst



def inputArguments():
    strRst = None
    args = None
    parser = ArgumentParser()
    parser.add_argument("modelname",help="File name of the model to test.")
    args = parser.parse_args()
    strRst = args.modelname
    del((args,parser))
    return strRst




if __name__ == "__main__":
    strMdlNme = inputArguments()
    dvc = device("cpu") # select computing devices.
    dataset = PtlFulShpPair("../../data/aapccTest3/") # select a split according to the trained model you choose.
    tsrMapSub = ptload("../../data/aapccTest3/mapSub.pth") # same as line 136.
    aryFce = npload("../smalFaces.npy")

    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,pin_memory=False)
    mdlbuf = ptload(strMdlNme,map_location=dvc)

    tplSmp = None
    sample = None
    tplOut = None
    tsrOut = None
    scaErStat1 = 0
    scaErStat2 = 0
    tplCD = None
    lstErStat3 = [0,0]
    scaSmpCnt = 0

    filSpirls = open("spiralae/spirals.pkl","rb")
    lstSpirls = pkload(filSpirls)
    filU = open("spiralae/U.pkl","rb")
    lstU = pkload(filU)

    modelEnc = PartialShapeEncoder().to(dvc)
    modelDec = None
    scaI = 0



    while scaI < 4:
        lstSpirls[scaI] = lstSpirls[scaI].to(dvc)
        lstU[scaI] = lstU[scaI].to(dvc)
        scaI += 1

    modelDec = SpiralDecoder([128, 64, 64, 32, 3], lstSpirls, 128, lstU, act="lrelu").to(dvc)

    filSpirls.close()
    filU.close()
    del(lstU,lstSpirls,filSpirls,filU,scaI)

    modelEnc.load_state_dict(mdlbuf["model_e"])
    modelDec.load_state_dict(mdlbuf["model_d"])
    del(mdlbuf)

    modelEnc.eval()
    modelDec.eval()
    set_grad_enabled(False)

    tsrMapSub = reshape(tsrMapSub,[tsrMapSub.size(0)//10,10])[:,0]

    for tplSmp in tqdm(enumerate(dataloader)):
        if not tsrMapSub[tplSmp[0]//5] == 2: # Aggregating quantitative results of Feline (0), Canine (1), Equine (2) and Bovine (3).
            continue
        sample = tplSmp[1].to(dvc)
        tplOut = modelEnc(sample.p,sample.batch,sample.num_graphs)
        tsrOut = reshape(modelDec(tplOut[1][:,0:64],tplOut[1][:,64:128]),[sample.num_graphs*3889,3])

        scaErStat1 += sum(mean(reshape(sqrt(sum((tsrOut-sample.r)**2,dim=1)),[sample.num_graphs,3889]),dim=1)).item()
        scaErStat2 += sample_wise_volerr(tsrOut,sample.r,aryFce)
        tplCD = sample_wise_cd(tsrOut,sample.r)
        lstErStat3[0] += tplCD[0]
        lstErStat3[1] += tplCD[1]
        scaSmpCnt += 1
    print(scaSmpCnt)
    print("PVE: {}.".format(scaErStat1/scaSmpCnt))
    print("CD: {}+{}={}.".format(lstErStat3[0]/scaSmpCnt,lstErStat3[1]/scaSmpCnt,
                                 (lstErStat3[0]+lstErStat3[1])/scaSmpCnt))
    print("Vol: {}.".format(scaErStat2/scaSmpCnt))
