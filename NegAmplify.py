import torch
import os
import os.path as osp
import GCL.losses as Loss
import GCL.losses as L
import GCL.augmentors as Aug
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikiCS, Amazon
import random
import pandas as pd
from torch.utils.data import random_split
torch.cuda.empty_cache()


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

def generate_negative_sample_masks(similarity_matrix, easy_percentage, hard_percentage, middle_percentage):
    num_anchors, num_samples = similarity_matrix.size()

    # Calculate the number of samples for each type based on the chosen percentages
    num_easy = int(num_samples * easy_percentage)
    num_hard = int(num_samples * hard_percentage)
    num_middle = int(num_samples * middle_percentage)

    # Sort the similarity matrix to get the indices for each type of negative sample
    sorted_indices = torch.argsort(similarity_matrix, dim=1)

    # Generate masks for each type of negative sample
    easy_mask = torch.zeros_like(similarity_matrix).float()
    hard_mask = torch.zeros_like(similarity_matrix).float()
    middle_mask = torch.zeros_like(similarity_matrix).float()

    easy_mask[torch.arange(num_anchors).unsqueeze(1), sorted_indices[:, :num_easy]] = True
    hard_mask[torch.arange(num_anchors).unsqueeze(1), sorted_indices[:, -num_hard:]] = True
    middle_mask[torch.arange(num_anchors).unsqueeze(1), sorted_indices[:, num_middle:num_middle+num_middle]] = True

    return easy_mask, hard_mask, middle_mask
def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()
def getMasks(anchor, sample,data,ad):
    assert anchor.size(0) == sample.size(0)
    num_nodes = anchor.size(0)  # getting the number of nodes
    device = anchor.device
    pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
    sim = _similarity(anchor, sample)  # _similarity() normalize the tensors as it is must do for cosine similarity.
    easy_mask, hard_mask, middle_mask = generate_negative_sample_masks(sim, ad[6],ad[7], ad[8])
    neg_mask= (easy_mask + hard_mask + middle_mask) -pos_mask
    return pos_mask, neg_mask
    
class InfoNCE():
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, tau):
        sim = _similarity(anchor, sample) / tau # _similarity() normalize the tensors as it is must do for cosine similarity.
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True)) # log compresses the large  values
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

def train(ad,encoder_model, contrast_model, data, optimizer):

    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    pos_mask1, neg_mask1 = getMasks(h1, h2, data, ad)
    pos_mask2, neg_mask2 = getMasks(h2, h1, data, ad)
    l1 = InfoNCE.compute(0, h1, h2, pos_mask1, neg_mask1, ad[10])
    l2 = InfoNCE.compute(0, h2, h1, pos_mask2, neg_mask2, ad[10])
    loss = (l1 + l2) * 0.5
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    torch_seed = 6521
    torch.manual_seed(torch_seed)
    r_seed= 9134
    random.seed(r_seed)
    Algo = "NegAmplify"
    d = "cuda:0"
    device = torch.device(d)
    dsName = 'Cora'
    path = osp.join(osp.expanduser('~'), 'datasets', 'Planetoid')
    print(path)
    dataset = Planetoid(path, name=dsName) 
    data = dataset[0].to(device)
    drop_prob = 0.3
    percentage = 1
    pf = str(drop_prob) + 'x' + str(percentage)

    ER1 =  0.45
    ER2 =  0.15
    FM1 =  0.35
    FM2 =  0.5
    taus =  0.4
    hyper = " " + str(ER1) + " " + str(FM1) + " " + str(ER2) + " " + str(FM2)
    print(hyper)
    outputFile = dsName + " " + Algo + hyper + ".csv" #
    if os.path.isfile(outputFile):
        df = pd.read_csv(outputFile, index_col=0)
    else:
        df = pd.DataFrame(columns=['TorchSeed', 'Dataset', 'Epochs', 'Result','TotalDrop', 'MicroF1', 'MacroF1', 'max_percentage','Tau', 'hyper'])

    for xws in range(0, 10):
        print("Round: ", xws)
        Oneseed = 6735
        twoseed = 1578
        torch.manual_seed(Oneseed)
        random.seed(twoseed)
        aug1 = Aug.Compose([Aug.EdgeRemoving(pe=ER1), Aug.FeatureMasking(pf=FM1)])
        aug2 = Aug.Compose([Aug.EdgeRemoving(pe=ER2), Aug.FeatureMasking(pf=FM2)])
        gconv = GConv(input_dim=dataset.num_features, hidden_dim=128, activation=torch.nn.ReLU, num_layers=2).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=128, proj_dim=128).to(device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=taus), mode='L2L', intraview_negs=True).to(device)
        optimizer = Adam(encoder_model.parameters(), lr=0.0005)

        epochs = 1201
        r = []
        r2 = []
        highestResults = (0, 0)
        losslist = []
        easy_percentage = 0.025
        hard_percentage = 0.025
        middle_percentage = 0.05
        max_percentage = 0.5
        ad = (0, 0, dsName, 0, 0, losslist, easy_percentage, hard_percentage, middle_percentage, max_percentage,taus)
        with tqdm(total=1200, desc='(T)') as pbar:
            for epoch in range(1, epochs):
                loss = train(ad,encoder_model, contrast_model, data, optimizer)
                losslist.append(loss)
                if len(losslist) >= 50 and len(losslist) % 20 == 0:
                    latest_loss = sum(losslist[-10:])
                    secondlatest_loss = sum(losslist[-20:-10])
                    if (latest_loss + 0.1) < (secondlatest_loss):  # it means model is learning ok
                        pass
                    elif (easy_percentage + hard_percentage + middle_percentage) < max_percentage:
                        # increase negative samples.
                        easy_percentage = easy_percentage + 0.01
                        hard_percentage = hard_percentage + 0.01
                        middle_percentage = middle_percentage + 0.02
                        print("increasing negative samples: Total: ",
                              (easy_percentage + hard_percentage + middle_percentage), len(losslist))
                ad = (epoch, 0, dsName, 0, 0, losslist, easy_percentage, hard_percentage, middle_percentage, max_percentage,taus)
                pbar.set_postfix({'loss': loss})
                pbar.update()
                if epoch % 50 == 0:
                    test_result = test(encoder_model, data)
                    r.append(test_result["micro_f1"])
                    r2.append(test_result["macro_f1"])
                    if highestResults[1] < test_result["micro_f1"]:
                        highestResults = (epoch, test_result["micro_f1"])
        max_value = max(r)
        max_value2 = max(r2)
        print(f'(E): Best test F1Mi={max_value:.4f}, F1Ma={max_value2:.4f}')
        values_to_add = {'TorchSeed': str(torch_seed)+" "+ str(Oneseed)+"  "+ str(twoseed) +" "+str(r_seed),
                         'Dataset': dsName, 'Epochs': highestResults[0],"Result": highestResults[1], 'TotalDrop': pf, 'MicroF1': max_value,
                         'MacroF1': max_value2,'max_percentage':max_percentage, 'Tau': taus, 'hyper':hyper}
        row_to_add = pd.Series(values_to_add)
        df = df.append(row_to_add, ignore_index=True)       
        df.to_csv(outputFile)


if __name__ == '__main__':
    main()

