`make_grid`, `ImageModule.show`, and `ImageModule.save` now interpret `n_row` as images per row, as documented.
Previously, it set the number of rows, so non-square grids had different dimensions and image positions.
The default uses the rounded-up square root of the batch size as the number of columns.
