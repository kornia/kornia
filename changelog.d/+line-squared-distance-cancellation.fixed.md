`ParametrizedLine.squared_distance` and `distance` no longer go negative or return NaN for points on or near the line.
They use the sum of squares of the perpendicular component again, row by row, instead of a cancelling identity. (#5016)
