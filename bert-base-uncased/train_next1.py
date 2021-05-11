import torch
p = torch.as_tensor(1.,device='cuda:0')
#p = tensor(1., device='cuda:0')
print(p)
print(p.squeeze())
q = p.cpu().detach().numpy().tolist()
print(type(q))
if isinstance(q, list) == False:
    print("hello")
print(q)