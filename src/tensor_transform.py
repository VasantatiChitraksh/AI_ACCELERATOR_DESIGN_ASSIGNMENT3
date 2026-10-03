import numpy as np

def flatten_tensor(tensor):
    """
    Flattens a BCHW tensor to B x (CHW)
    tensor: numpy array of shape (B, C, H, W)
    """
    B, C, H, W = tensor.shape
    flattened = np.zeros((B, C * H * W), dtype=tensor.dtype)
    
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    idx = c * (H * W) + h * W + w
                    flattened[b, idx] = tensor[b, c, h, w]
                    
    return flattened

def reconstruct_tensor(flattened, original_shape):
    """
    Reconstructs a B x (CHW) tensor back to B x C x H x W
    flattened: numpy array of shape (B, CHW)
    original_shape: tuple (B, C, H, W)
    """
    B, C, H, W = original_shape
    reconstructed = np.zeros((B, C, H, W), dtype=flattened.dtype)
    
    for b in range(B):
        for idx in range(C * H * W):
            w = idx % W
            h = (idx // W) % H
            c = idx // (H * W)
            reconstructed[b, c, h, w] = flattened[b, idx]
            
    return reconstructed
