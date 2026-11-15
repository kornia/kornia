Fix LoFTR linear attention erasing float16 outputs and value gradients when its unnormalized sums overflow at
ordinary image sizes. Half-precision attention now computes in float32, outside autocast in eager and compiled
calls, and returns the query dtype. Fully masked key sequences also return finite zeros.
