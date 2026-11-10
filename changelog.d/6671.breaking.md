`posterize` now raises for `bits` outside `[0, 8]`, given as an int or a tensor. It used to return a black image for such values.
