import torch
from models.cifar import wrn_reparametrized

SEED = 42
torch.manual_seed(SEED)
x = torch.randn(10, 3, 100, 100)

NC = 10
net_orig = wrn_reparametrized(depth=28, widen_factor=10, num_classes=NC, reparametrized=False)
net_reparam = wrn_reparametrized(depth=28, widen_factor=10, num_classes=NC, reparametrized=True)
net_orig_dict = net_orig.state_dict()

for k, v in net_reparam.state_dict().items():
    if "conv1.weight" in k or "conv2.weight" in k or "reparametrized" in k:
        continue

    if "conv1_original_weight" in k:
        k_ = k.replace("conv1_original_weight", "conv1.weight")
        net_orig_dict[k_].copy_(v.data)

    if "conv2_original_weight" in k:
        k_ = k.replace("conv2_original_weight", "conv2.weight")
        net_orig_dict[k_].copy_(v.data)

    if k not in net_orig_dict:
        continue

    net_orig_dict[k].copy_(v.data)

print("distance between orig and reparametrized", torch.norm(net_orig(x) - net_reparam(x)))
torch.save(net_reparam.state_dict(), f"wrn-reparametrized-seed{SEED}.pth")
print(f"saved reparametrized state_dict (with original weights included) to wrn-reparametrized-seed{SEED}.pth")