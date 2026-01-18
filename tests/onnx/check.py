import sys
sys.path.insert(0, r'C:\Users\lenovo\KORNIAAA\kornia\tests\onnx')
import test_resize_onnx
test_resize_onnx.test_resize_dynamo_with_binding()
print('Test 1 passed')
test_resize_onnx.test_resize_upscale_dynamo()
print('Test 2 passed')
test_resize_onnx.test_resize_downscale_dynamo()
print('Test 3 passed')
test_resize_onnx.test_resize_nearest_dynamo()
print('Test 4 passed')
print('ALL TESTS PASSED')