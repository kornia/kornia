`spatial_gradient` and `sobel` differentiate an integer image in float32 instead of raising, matching
`spatial_gradient3d`. In the input dtype the Sobel taps truncated and uint8 differences wrapped.
