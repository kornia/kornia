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

import pytest
import torch
from kornia.depth_anything_v3.common import Attention,LayerScale,MLP,DropPath,Block
#======================= TESTS FOR ATTENTION =======================
def test_attention_forward_shape():
    batch_size, seq_len, dim = 2, 14, 64
    nb_head = 8
    model = Attention(dim=dim, nb_head=nb_head)
    
    x = torch.randn(batch_size, seq_len, dim)
    out = model(x)
    
    assert out.shape == (batch_size, seq_len, dim)
    assert out.dtype == torch.float32

def test_attention_edge_cases():
    with pytest.raises(ValueError, match="dim must be > 0"):
        Attention(dim=0, nb_head=4)
        
    with pytest.raises(ValueError, match="nb_head must be > 0"):
        Attention(dim=64, nb_head=0)
        
    with pytest.raises(ValueError, match="must be divisible"):
        Attention(dim=64, nb_head=3)

def test_attention_backprop():
    batch_size, seq_len, dim = 2, 14, 64
    nb_head = 4
    model = Attention(dim=dim, nb_head=nb_head)
    
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    
    out = model(x)
    #fake loss for the test
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    
    for name, param in model.named_parameters():
        #make sure each parameters has a gradeitn
        assert param.grad is not None
        #make sure the gradient isn't full of zeros
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad),atol=1e-12)


#======================= TESTS FOR LAYERSCALE CLASS =======================
def test_layerscale_forward_shape():
    batch_size, seq_len, dim = 2, 14, 64
    model = LayerScale(dim=dim)
    
    x = torch.randn(batch_size, seq_len, dim)
    out = model(x)
    #make sur the output dimensions are good
    assert out.shape == (batch_size, seq_len, dim)

def test_layerscale_initialization():
    dim = 64
    init_val = 1e-5
    model = LayerScale(dim=dim, init_value=init_val)
    
    x = torch.ones(1, 1, dim)
    out = model(x)
    
    expected_out = torch.ones(1, 1, dim) * init_val
    #make sur the layerscale compute the good value
    assert torch.allclose(out, expected_out)

def test_layerscale_backprop():
    batch_size, seq_len, dim = 2, 14, 64
    model = LayerScale(dim=dim)
    
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    out = model(x)
    
    loss = out.sum()
    loss.backward()
    #make sure grad are not None and not equals to zeros
    assert x.grad is not None
    assert model.gamma.grad is not None
    assert not torch.allclose(model.gamma.grad, torch.zeros_like(model.gamma.grad))

#======================= TESTS FOR MLP =======================
def test_mlp_forward_shape():
    batch_size, seq_len, in_features = 2, 14, 64
    x = torch.randn(batch_size, seq_len, in_features)
    
    # case where (dim_in_features == dim_hidden_features == dim_out_features)
    model_default = MLP(dim_in_f=in_features)
    out_default = model_default(x)
    assert out_default.shape == (batch_size, seq_len, in_features)
    
    # case where dimensions are personalized
    hidden_features = in_features * 4
    out_features = 32
    model_custom = MLP(dim_in_f=in_features, dim_hidden_f=hidden_features, dim_out_f=out_features)
    out_custom = model_custom(x)
    assert out_custom.shape == (batch_size, seq_len, out_features)

def test_mlp_backprop():
    batch_size, seq_len, in_features = 2, 14, 64
    model = MLP(dim_in_f=in_features, dim_hidden_f=in_features * 4)
    
    x = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
    out = model(x)
    
    loss = out.sum()
    loss.backward()
    #make sure grad and paramters grad are not None and not null
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
#======================= TESTS FOR DROPPATH =======================
def test_droppath_forward_shape():
    x = torch.randn(2, 14, 64)
    model = DropPath(drop_prob=0.2)
    #test with train mode
    model.train()
    out_train = model(x)
    #check if the shape stay good
    assert out_train.shape == x.shape

def test_droppath_inference_identity():
    x = torch.randn(2, 14, 64)
    model = DropPath(drop_prob=0.5)
    model.eval()
    out = model(x)
    #maje sure dropath doesn't act in eval mode
    assert torch.equal(out, x)

def test_droppath_zero_prob():
    x = torch.randn(2, 14, 64)
    model = DropPath(drop_prob=0.0)
    model.train()
    out = model(x)
    #make sure dropath doesn't act if drop_prob==0
    assert torch.equal(out, x)

#======================= TESTS FOR BLOCK =======================
def test_block_forward_shape():
    batch_size, seq_len, dim = 2, 14, 64
    nb_head = 4
    
    model = Block(dim=dim, nb_head=nb_head,dim_hidden_f=dim*2)
    x = torch.randn(batch_size, seq_len, dim)
    out = model(x)
    #make sure the dimension stay good
    assert out.shape == (batch_size, seq_len, dim)

def test_block_backprop():
    batch_size, seq_len, dim = 2, 14, 64
    nb_head = 4
    
    model = Block(dim=dim, nb_head=nb_head,dim_hidden_f=dim*2)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    out = model(x)
    
    loss = out.sum()
    loss.backward()
    
    # make sure the gradient pass through the whole block
    assert x.grad is not None
    
    #make sur attention layer , layerscale and mlp get a part of the gradient
    assert model.attention_layer.w_q.weight.grad is not None
    assert model.mlp_layer.fc1.weight.grad is not None
    assert model.mlp_layer.fc2.weight.grad is not None
    assert model.layerscale1.gamma.grad is not None