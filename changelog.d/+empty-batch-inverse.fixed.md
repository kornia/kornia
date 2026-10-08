Fix an IndexError from padding[0] when replaying saved parameters for an empty RandomCrop mask inverse
without a cached transformation matrix. Part of #4429. Other empty-batch inverse paths remain unchanged.
