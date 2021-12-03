from models.dataset import Dataset
from pyhocon import ConfigFactory
import torch

conf_path = "./confs/womask.conf"
case = "dtu24"

f = open(conf_path)
conf_text = f.read()
conf_text = conf_text.replace('CASE_NAME', case)
f.close()

conf = ConfigFactory.parse_string(conf_text)
conf['dataset.data_dir'] = conf['dataset.data_dir'].replace('CASE_NAME', case)

dataset = Dataset(conf['dataset'])

print(dataset.pose_all[:, :3, 3])

"""
# pixels_x = torch.randint(low=0, high=dataset.W, size=[2])
# pixels_y = torch.randint(low=0, high=dataset.H, size=[2])
pixels_x = torch.Tensor([1538, 714]).long()
pixels_y = torch.Tensor([255, 343]).long()
print(pixels_x, pixels_y)

color = dataset.images[0][(pixels_y, pixels_x)]    # batch_size, 3
mask = dataset.masks[0][(pixels_y, pixels_x)]      # batch_size, 3
p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
p = torch.matmul(dataset.intrinsics_all_inv[0, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3

rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
rays_v = torch.matmul(dataset.pose_all[0, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
rays_o = dataset.pose_all[0, None, :3, 3].expand(rays_v.shape) # batch_size, 3

print("v:", rays_v)
print("o:", rays_o)
"""