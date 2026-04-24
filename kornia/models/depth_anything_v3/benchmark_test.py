
import torch 
from kornia.models.depth_anything_v3.common import Block
import torch.utils.benchmark as benchmark

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, Seq, D = 32, 128, 256
    nb_head = 8
    dim_hidden_f = D * 4
    
    #create dataset
    x = torch.randn(B, Seq, D, device=device)


    my_block = Block(
        dim=D, nb_head=nb_head, dim_hidden_f=dim_hidden_f
    ).to(device)

    official_block = torch.nn.TransformerEncoderLayer(
        d_model=D, nhead=nb_head, dim_feedforward=dim_hidden_f
    ).to(device)

    #put eval mode
    my_block.eval()
    official_block.eval()

    t0 = benchmark.Timer(
        stmt='my_block(x)',
        globals={'my_block': my_block, 'x': x},
    )
    t1 = benchmark.Timer(
        stmt='official_block(x)',
        globals={'official_block': official_block, 'x': x},
    )
    
    #do 100 times the operation 
    print("time for common.py block",t0.timeit(100))
    print("time for torch block :",t1.timeit(100))


if __name__ == '__main__':
    run_benchmark()