import torch

a_list = [1, 2, 3, 4, 5]
b_list = [1, 2, 3, 4, 5]

a_torch = torch.tensor(a_list, dtype=torch.uint8)
b_torch = torch.tensor(b_list, dtype=torch.uint8)

qa_torch = torch.quantize_per_tensor(a_torch.to(torch.float), scale=1.0, zero_point=0, dtype=torch.quint8)
qb_torch = torch.quantize_per_tensor(b_torch.to(torch.float), scale=1.0, zero_point=0, dtype=torch.quint8)

print(f'- qa_torch={qa_torch}') 
print(f'- qb_torch={qb_torch}')
# - qa_torch=tensor([212., 181.,   3.,  89., 147.], size=(5,), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=1.0, zero_point=0)
# - qa_torch.int_repr=tensor([212, 181,   3,  89, 147], dtype=torch.uint8)
# - qb_torch=tensor([220., 207.,   3., 228., 172.], size=(5,), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=1.0, zero_point=0)
# - qb_torch.int_repr=tensor([220, 207,   3, 228, 172], dtype=torch.uint8)

qc_torch = torch.ops.quantized.mul(qa_torch, qb_torch, scale=1.0, zero_point=0)
print("________")
print(f'- qc_torch={qc_torch}')
# - qc_torch=tensor([255., 255.,   9., 255., 255.], size=(5,), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=1.0, zero_point=0)
# - qc_torch.int_repr=tensor([255, 255,   9, 255, 255], dtype=torch.uint8)
