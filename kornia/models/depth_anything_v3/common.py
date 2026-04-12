# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations
import torch
from kornia.models.common import DropPath

class Attention(torch.nn.Module):
    """
    Multi head attentions layer
    """
    def __init__(self, dim:int, nb_head :int, bias_for_qkv:bool =True)->None:
        super().__init__()
        if dim<=0:
            raise ValueError("dim must be > 0")
        if nb_head<=0:
            raise ValueError("nb_head must be > 0")
        self.dim=dim 
        self.nb_head=nb_head 
        self.head_dim = dim//nb_head 
        if self.head_dim*nb_head!=dim:
            raise ValueError(f"dim{dim} must be divisible by the number of heads {nb_head}")
        #weights layers for queries,keys and values
        self.w_q=torch.nn.Linear(dim,dim,bias=bias_for_qkv)
        self.w_k=torch.nn.Linear(dim,dim,bias=bias_for_qkv)
        self.w_v=torch.nn.Linear(dim,dim,bias=bias_for_qkv)
        #projection layer for the output 
        self.projection_layer = torch.nn.Linear(dim,dim)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        B,C,D=x.shape
        #output after the queries, keys and values layers 
        q_out=self.w_q(x)
        k_out=self.w_k(x)
        v_out=self.w_v(x) 

        #reshape to separate the heads with the new dimensions : (B,C,nb_head,head_dim)
        q_out=q_out.view(B,C,self.nb_head,self.head_dim)
        k_out=k_out.view(B,C,self.nb_head,self.head_dim)
        v_out=v_out.view(B,C,self.nb_head,self.head_dim)

        #transpose the matrix, for the dimensions to be (B,nb_head,C,head_dim) 
        q_out=q_out.transpose(1,2)
        k_out=k_out.transpose(1,2)
        v_out=v_out.transpose(1,2)

        #computation of the attention
        attention = torch.nn.functional.scaled_dot_product_attention(q_out,k_out,v_out)

        #transpose the matrix to come back to the initial dimensions 
        attention=attention.transpose(1,2)
        #reshape attention to come back to the initial dimensions 
        attention=attention.reshape(B,C,D)

        output=self.projection_layer(attention)
        return output
    
class LayerScale(torch.nn.Module):
    """LayerScale module.
    
    Multiplies the input by a learnable diagonal matrix"""
    def __init__(self,dim:int, init_value:float = 1e-5, inplace:bool = False)->None:
        super().__init__()
        if dim <=0 : 
            raise ValueError("dim must be > 0")
        self.inplace=inplace 
        self.gamma=torch.nn.Parameter(init_value*torch.ones(dim))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.inplace : 
            return x.mul_(self.gamma)
        return x*self.gamma
class MLP(torch.nn.Module): 
    """Multilayer perceptron module"""
    def __init__(self, dim_in_f:int,dim_hidden_f:int|None = None,dim_out_f:int|None = None)->None:
        super().__init__()
        if dim_hidden_f is None:
            dim_hidden_f=dim_in_f 
        elif dim_hidden_f<=0:
            raise ValueError("dimension of the hidden layer must be > 0")
        if dim_out_f is None:
            dim_out_f=dim_in_f 
        elif dim_out_f<=0:
            raise ValueError("dimension of the output must be > 0")
        if dim_in_f<=0:
            raise ValueError("dimension of the input must be > 0")
        
        #first linear layer 
        self.fc1=torch.nn.Linear(dim_in_f,dim_hidden_f)
        #activation layer, GELU Is the more stable for Attention layers 
        self.acti=torch.nn.GELU()
        #second linear layer 
        self.fc2=torch.nn.Linear(dim_hidden_f,dim_out_f) 
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x=self.fc1(x)
        x=self.acti(x)
        x=self.fc2(x)
        return x
class Block(torch.nn.Module):
    """Vision Transformer Block
    LayerNormalisation->Attention->LayerScale->DropPath->LayerNorm->MLP->LayerScale->Dropath
    """
    def __init__(self,dim:int, nb_head:int, dim_hidden_f:int, bias_for_qkv:bool = True, drop_prob : float = 0.0, init_value:float = 1e-5, scale_by_keep:bool = True)->None:
        super().__init__()
        
        self.norm_layer1 = torch.nn.LayerNorm(dim)
        self.attention_layer = Attention(dim=dim, nb_head=nb_head, bias_for_qkv=bias_for_qkv)
        self.layerscale1= LayerScale(dim=dim, init_value=init_value)
        self.drop_path1 = DropPath(drop_prob=drop_prob,scale_by_keep=scale_by_keep)
        
        self.norm_layer2 = torch.nn.LayerNorm(dim)
        self.mlp_layer = MLP(dim_in_f=dim, dim_hidden_f=dim_hidden_f)
        self.layerscale2 = LayerScale(dim=dim, init_value=init_value)
        self.drop_path2 = DropPath(drop_prob=drop_prob,scale_by_keep=scale_by_keep)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        #attention path
        x = x + self.drop_path1(self.layerscale1(self.attention_layer(self.norm_layer1(x))))
        #mlp path
        x = x + self.drop_path2(self.layerscale2(self.mlp_layer(self.norm_layer2(x))))
        return x        