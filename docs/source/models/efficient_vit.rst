EfficientViT
============

.. card::
    :link: https://arxiv.org/abs/2205.14756

    **EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction**
    ^^^
    **Abstract:** High-resolution dense prediction enables many appealing real-world applications, such as computational photography, autonomous driving, etc. However, the vast computational cost makes deploying state-of-the-art high-resolution dense prediction models on hardware devices difficult. This work presents EfficientViT, a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel convolution, or complicated topology structure to obtain good performances, our multi-scale linear attention achieves the global receptive field and multi-scale learning (two desirable features for high-resolution dense prediction) with only lightweight and hardware-efficient operations. As such, EfficientViT delivers remarkable performance gains over previous state-of-the-art models with significant speedup on diverse hardware platforms, including mobile CPU, edge GPU, and cloud GPU. Without performance loss on Cityscapes, our EfficientViT provides up to 13.9x and 6.2x GPU latency reduction over SegFormer and SegNeXt, respectively. For super-resolution, EfficientViT delivers up to 6.4x speedup over Restormer while providing 0.11dB gain in PSNR. For Segment Anything, EfficientViT delivers similar zero-shot image segmentation quality as ViT-Huge with 84x higher throughput on GPU. Code: this https URL.

    **Tasks:** Classification, Segmentation, Detection

    **Licence:** Apache 2.0

    +++
    **Authors:** Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
