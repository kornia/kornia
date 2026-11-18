`guided_blur` reduces peak eager memory for multichannel guidance by releasing unused second-moment image buffers before the linear solve and coefficient filtering.
