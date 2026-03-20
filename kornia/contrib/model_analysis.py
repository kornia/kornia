import torch

def analyze_model_behavior(model, image, augmentation, layers=("layer1",)):
    """
    Analyze how model internal representations change under augmentations.

    This utility applies an augmentation to an input image and compares
    intermediate feature maps across specified layers.

    Args:
        model: PyTorch model
        image: Input tensor of shape (B, C, H, W)
        augmentation: Kornia augmentation module
        layers: Tuple of layer names to analyze

    Returns:
        dict: Mapping layer name → mean absolute difference
    """
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    handles = []

    # Register hooks
    for layer_name in layers:
        layer = dict([*model.named_modules()])[layer_name]
        handles.append(layer.register_forward_hook(hook_fn(layer_name)))

    with torch.no_grad():
        # Original pass
        features.clear()
        _ = model(image)
        orig_feats = {k: v.clone() for k, v in features.items()}

        # Augmented pass
        features.clear()
        aug_image = augmentation(image)
        _ = model(aug_image)
        aug_feats = {k: v.clone() for k, v in features.items()}

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute differences
    diffs = {}
    for k in orig_feats:
        diffs[k] = (orig_feats[k] - aug_feats[k]).abs().mean().item()

    return diffs