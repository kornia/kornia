`js_div_loss_2d` now evaluates its mixture in log space, keeping Gaussian-heatmap losses and coordinate gradients finite when halving tiny float32 or float64 probabilities would underflow.
