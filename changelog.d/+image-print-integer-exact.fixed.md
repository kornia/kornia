`kornia.image.image_to_string` and `print_image` match an integer image's pixels at their exact 0–255 values on every
device, as documented. The image was divided by 255 and multiplied back, which is exact on CPU but moved 126 of the 256
`uint8` levels off their integer on CUDA, where a division by a scalar is a multiplication by its reciprocal. A CUDA
image could therefore print some pixels in a different colour than its CPU copy. CPU output is unchanged.
