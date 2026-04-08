import torch
class Attention(torch.nn.Module):
    """
    Multi head attentions layer
    """
    def __init__(self, dim:int, nb_heads :int, bias_for_qkv=True)->None:
        super().__init__()
        if dim<=0:
            raise ValueError("dim must be > 0")
        if nb_heads<=0:
            raise ValueError("nb_heads must be > 0")
        self.dim=dim 
        self.nb_heads=nb_heads 
        self.head_dim = dim//nb_heads 
        if self.head_dim*nb_heads!=dim:
            raise ValueError(f"dim{dim} must be divisible by the number of heads {nb_heads}")
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

        #reshape to separate the heads with the new dimensions : (B,C,nb_heads,head_dim)
        q_out=q_out.view(B,C,self.nb_heads,self.head_dim)
        k_out=k_out.view(B,C,self.nb_heads,self.head_dim)
        v_out=v_out.view(B,C,self.nb_heads,self.head_dim)

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