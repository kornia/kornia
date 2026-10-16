`draw_line` accepts a 0-d scalar color for single-channel images instead of raising `IndexError` on `color.size(0)`.
