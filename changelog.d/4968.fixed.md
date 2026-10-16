`kornia.geometry.euclidean_distance` now returns the exact Euclidean distance. The previous form
added `eps` (1e-6) inside the square root, which biased every result upwards and floored it at
`sqrt(eps)` = 1e-3: coincident points read 0.001 and a true distance of 1e-4 read 1.005e-3
(+905 %). The singular point is guarded with `torch.where` instead, so coincident points return
exactly 0 with a finite (zero) gradient, as `torch.linalg.norm` does. The `eps` argument is now
unused and is kept for backward compatibility.
