import torch.nn as nn
import torch
class DiabetesClassifier(nn.Module):
    def __init__(self,embd_szs,n_cont,layers=[150,150,150],output_dim=2):
        super().__init__()
        self.embd_list=nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in embd_szs])
        self.embd_dropout=nn.Dropout(0.2)

        x=sum([nf for ni,nf in embd_szs])+n_cont

        layer_list=[]
        layer_list.append(nn.BatchNorm1d(x))
        for i in layers:
            layer_list.append(nn.Linear(x,i))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Dropout(0.4))
            x=i
        layer_list.append(nn.Linear(layers[-1],output_dim))
        self.layers=nn.Sequential(*layer_list)

    def forward(self,x_cont,x_cat):
        embeddings=[]
        for i,e in enumerate(self.embd_list):
            embeddings.append(e(x_cat[:,i]))

        x_cat=torch.cat(embeddings,1)
        x_cat=self.embd_dropout(x_cat)
        input_features=torch.cat([x_cont,x_cat],1)
        return self.layers(torch.tensor(input_features,dtype=torch.float))
