import time
import numpy as np
import torch
import torch.nn as nn
import torch.quantization

M, K, N = 2048, 2048, 2048
a_fp = torch.randn(M, K) # Look into creating qint8 tensors instead of converting
b_fp =  torch.randn(K, N)
scale, zero_point = 1, 0
a_q = torch.quantize_per_tensor(a_fp, scale=scale, zero_point=zero_point, dtype=torch.qint8)
b_q = torch.quantize_per_tensor(b_fp, scale=scale, zero_point=zero_point, dtype=torch.qint8)

print(a_q)
print(b_q)
q_f = torch.ao.nn.quantized.QFunctional()

start = time.time()
for _ in range(1):
    c_q = q_f.matmul(a_q, b_q)
    print(c_q)
end = time.time()
print("PyTorch quantized matmul time:", end - start)

a_np = a_q.int_repr().numpy().astype(np.int8)
b_np = b_q.int_repr().numpy().astype(np.int8)
print(a_np)
print(b_np)
start = time.time()
for _ in range(1):
    c_np = np.matmul(a_np, b_np)
    print(c_np)
end = time.time()
print("NumPy int8 matmul time:", end - start)

