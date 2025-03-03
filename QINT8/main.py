import torch
import torch.nn as nn
import torch.quantization

# Create example data
M, K, N = 128, 64, 32
a_fp = torch.randn(M, K) # (128, 64)
b_fp = torch.randn(K, N) # (64, 32)

scale, zero_point = 0.1, 0

a_q = torch.quantize_per_tensor(a_fp, scale=scale, zero_point=zero_point, dtype=torch.qint8)
b_q = torch.quantize_per_tensor(b_fp, scale=scale, zero_point=zero_point, dtype=torch.qint8)

q_f = torch.ao.nn.quantized.QFunctional()

c_q = q_f.matmul(a_q, b_q)
print(c_q)
