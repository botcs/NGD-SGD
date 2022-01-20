import torch
from models.cifar import vgg_reparametrized

SEED = 42
torch.manual_seed(SEED)
net = vgg_reparametrized.vgg16(num_classes=10, reparametrized=False)
x = torch.randn(10, 3, 32, 32)
y = net(x)
net.reparametrized(True)
yp = net(x)

print("distance between orig and reparametrized", torch.norm(y-yp))
torch.save(net.state_dict(), f"vgg-reparametrized-seed{SEED}.pth")
print(f"saved reparametrized state_dict (with original weights included) to vgg-reparametrized-seed{SEED}.pth")