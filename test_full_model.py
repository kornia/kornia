import torch
import sys
import os

# Force Python to look in the current directory for 'kornia'
sys.path.append(os.getcwd())

print("🔄 Attempting to import from kornia.models.qwen25...")

try:
    from kornia.models.qwen25.qwen2_vl import Qwen2VLVisionTransformer
    print("✅ Import Successful!")

    # 1. Instantiate the Full Model
    print("🔄 Instantiating Model...")
    model = Qwen2VLVisionTransformer(
        embed_dim=128, 
        depth=2, 
        num_heads=4, 
        patch_size=14
    )
    print("✅ Model Created.")

    # 2. Create Dummy Input
    # Shape: [Batch, SeqLen, EmbedDim]
    x = torch.randn(1, 16, 128)
    
    # 3. Run Forward Pass
    print("🔄 Running Forward Pass...")
    out = model(x)
    
    print(f"✅ Success! Output Shape: {out.shape}")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   (Did you create the __init__.py file in kornia/models/qwen25/ ?)")
except Exception as e:
    print(f"❌ Runtime Error: {e}")