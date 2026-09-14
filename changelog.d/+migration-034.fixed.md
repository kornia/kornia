`rad2deg` and `deg2rad` now handle integer tensor inputs correctly and preserve
float64 precision. `angle_to_rotation_matrix` inherits the corrected conversion,
while the implementation preserves ONNX export compatibility. (#4358)
