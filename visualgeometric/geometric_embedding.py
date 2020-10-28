import time
import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

import torch_geometric.data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

'''
Network to extract a simple embedding from a graph (normalized to 1), can be used to score the similarity of multiple graphs
'''
class GeometricEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(GeometricEmbedding, self).__init__()

        self.embedding_dim=embedding_dim

        #Graph layers
        self.conv1 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv2 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv3 = GCNConv(self.embedding_dim, self.embedding_dim)

        self.node_embedding=torch.nn.Embedding(30, self.embedding_dim) #30 should be enough
        self.node_embedding.requires_grad_(False)

    def forward(self, graphs):
        #x, edges, edge_attr, batch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        
        x = self.node_embedding(graphs.x) #CARE: is this ok? X seems to be simply stacked
        edges=graphs.edge_index
        # edge_attr=graphs.edge_attr
        batch=graphs.batch

        x = self.conv1(x, edges)
        x = F.relu(x)
        x = self.conv2(x, edges)
        x = F.relu(x)
        x = self.conv3(x, edges)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x= x/torch.norm(x, dim=1,keepdim=True) #Norm output
        
        return x

class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s): #Norming the input (as in paper) is actually not helpful
        im=im/torch.norm(im,dim=1,keepdim=True)
        s=s/torch.norm(s,dim=1,keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        #print(scores)
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return (cost_s.sum() + cost_im.sum()) / len(im) #Take mean for batch-size stability             