import torch
class Attention(torch.nn.Module):
    def __init__(self, dim:int, nb_heads :int, bias_for_qkv=True)->None:
        super().__init__()
        self.dim=dim 
        self.nb_heads=nb_heads 
        self.head_dim = dim//nb_heads 
        if self.head_dim*nb_heads!=dim:
            raise ValueError(f"dim{dim} must be divisible by the number of heads {nb_heads}")
        #weights layers for queries,keys and values
        self.w_q=torch.nn.linear(dim,dim,bias=bias_for_qkv)
        self.w_k=torch.nn.linear(dim,dim,bias=bias_for_qkv)
        self.w_v=torch.nn.linear(dim,dim,bias=bias_for_qkv)
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