import time
import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
import torchvision.models

import torch_geometric.data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

from retrieval.netvlad import NetVLAD

def create_image_model_resnet_18():
    model=torchvision.models.resnet18(pretrained=True)
    extractor = torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4)    
    return extractor

class VisualGraphEmbeddingCombined(torch.nn.Module):
    def __init__(self,image_model, embedding_dim):
        super(VisualGraphEmbeddingCombined, self).__init__()

        self.embedding_dim=embedding_dim

        #Graph layers
        self.conv1 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv2 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv3 = GCNConv(self.embedding_dim, self.embedding_dim)

        self.node_embedding=torch.nn.Embedding(30, self.embedding_dim) #30 should be enough
        self.node_embedding.requires_grad_(False) # Performance proved better w/o training the Embedding ✓

        self.image_model=image_model
        self.image_model.requires_grad_(False)
        self.image_model.eval()
        
        self.netvlad=NetVLAD(num_clusters=8, dim=512, alpha=10.0) #Best determined parameters so far
        self.image_dim=self.netvlad.dim * self.netvlad.num_clusters
        assert self.image_dim==4096     

        self.W_combine=torch.nn.Linear(self.image_dim+self.embedding_dim,self.embedding_dim,bias=True)

    def forward(self, images, graphs):
        #assert len(graphs)==len(images)

        out_images=self.encode_images(images)
        assert out_images.shape[1]==self.image_dim
        out_graphs=self.encode_graphs(graphs)
        assert out_graphs.shape[1]==self.embedding_dim

        assert out_images.shape[0]==out_graphs.shape[0]

        out_combined=self.W_combine( torch.cat((out_images, out_graphs), dim=1) ) #Concatenate along dim1 (4096+1024)
        out_combined = F.normalize(out_combined, p=2, dim=1)

        return out_combined
        
    def encode_images(self, images):
        assert len(images.shape)==4 #Expect a batch of images
        x=self.image_model(images)
        x=self.netvlad(x)

        return x

    def encode_graphs(self, graphs):
        #x, edges, edge_attr, batch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        
        x = self.node_embedding(graphs.x) #CARE: is this ok? X seems to be simply stacked
        edges=graphs.edge_index
        edge_attr=graphs.edge_attr
        batch=graphs.batch

        x = self.conv1(x, edges, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edges, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edges, edge_attr)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.normalize(x, p=2, dim=1)
        #TODO: normalize here?! (NetVLAD normalizes, too...)
        
        return x  

#Alternative version, receives a fully-trained NV-FC-model, only trains GE and FC layers
class VisualGraphEmbeddingCombinedPT(torch.nn.Module):
    def __init__(self,image_model, embedding_dim):
        super(VisualGraphEmbeddingCombinedPT, self).__init__()

        self.embedding_dim=embedding_dim

        #Graph layers
        self.conv1 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv2 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv3 = GCNConv(self.embedding_dim, self.embedding_dim)

        self.node_embedding=torch.nn.Embedding(30, self.embedding_dim) #30 should be enough
        self.node_embedding.requires_grad_(False) # Performance proved better w/o training the Embedding ✓

        self.image_model=image_model
        self.image_model.requires_grad_(False)
        self.image_model.eval()
        
        self.image_dim=self.image_model.netvlad.dim * self.image_model.netvlad.num_clusters
        assert self.image_dim==4096     

        self.W_combine=torch.nn.Linear(self.image_dim+self.embedding_dim,self.embedding_dim,bias=True)

    def forward(self, images, graphs):
        #assert len(graphs)==len(images)

        out_images=self.encode_images(images)
        assert out_images.shape[1]==self.image_dim
        out_graphs=self.encode_graphs(graphs)
        assert out_graphs.shape[1]==self.embedding_dim

        assert out_images.shape[0]==out_graphs.shape[0]

        out_combined=self.W_combine( torch.cat((out_images, out_graphs), dim=1) ) #Concatenate along dim1 (4096+1024)
        out_combined = F.normalize(out_combined, p=2, dim=1)

        return out_combined
        
    def encode_images(self, images):
        assert len(images.shape)==4 #Expect a batch of images
        x=self.image_model(images)
        return x

    def encode_graphs(self, graphs):
        #x, edges, edge_attr, batch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        
        x = self.node_embedding(graphs.x) #CARE: is this ok? X seems to be simply stacked
        edges=graphs.edge_index
        edge_attr=graphs.edge_attr
        batch=graphs.batch

        x = self.conv1(x, edges, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edges, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edges, edge_attr)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.normalize(x, p=2, dim=1)
        #TODO: normalize here?! (NetVLAD normalizes, too...)
        
        return x          