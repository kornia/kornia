import pytest
import torch
from common import Attention,LayerScale,MLP
def test_attention_forward_shape():
    batch_size, seq_len, dim = 2, 14, 64
    nb_heads = 8
    model = Attention(dim=dim, nb_heads=nb_heads)
    
    x = torch.randn(batch_size, seq_len, dim)
    out = model(x)
    
    assert out.shape == (batch_size, seq_len, dim)
    assert out.dtype == torch.float32

def test_attention_edge_cases():
    with pytest.raises(ValueError, match="dim must be > 0"):
        Attention(dim=0, nb_heads=4)
        
    with pytest.raises(ValueError, match="nb_heads must be > 0"):
        Attention(dim=64, nb_heads=0)
        
    with pytest.raises(ValueError, match="must be divisible"):
        Attention(dim=64, nb_heads=3)

def test_attention_backprop():
    batch_size, seq_len, dim = 2, 14, 64
    nb_heads = 4
    model = Attention(dim=dim, nb_heads=nb_heads)
    
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


#TESTS FOR LAYERSCALE CLASS
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

#TESTS FOR MLP
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