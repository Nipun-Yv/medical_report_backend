import torch.nn as nn
class ANNClassifier(nn.Module):
    def __init__(self,input_features,output_features,layers=[50,50]):
        super().__init__()
        layer_list=[]
        init=input_features
        for i in layers:
            layer_list.append(nn.Linear(init,i))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(0.4))
            init=i
        layer_list.append(nn.Linear(layers[-1],output_features))
        self.layer_list=nn.Sequential(*layer_list)
    def forward(self,X):
        return self.layer_list(X)