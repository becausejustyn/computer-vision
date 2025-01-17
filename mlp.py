import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

train_data = datasets.MNIST(root='./data', train=True, download=True)
test_data = datasets.MNIST(root='./data', train=False, download=True)

def maybe_cuda(x):
  if torch.cuda.is_available():
    return x.cuda()
  return x

Xtr = maybe_cuda(train_data.data.reshape(-1, 28*28).to(torch.float32) / 127.5 - 1.0)
Xte = maybe_cuda(test_data.data.reshape(-1, 28*28).to(torch.float32) / 127.5 - 1.0)
Ytr = maybe_cuda(train_data.targets)
Yte = maybe_cuda(test_data.targets)

xent = nn.CrossEntropyLoss()
model = maybe_cuda(nn.Linear(28*28, 10, bias=False))
opt = torch.optim.Adam(model.parameters(), lr=0.001)

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28*28, 256)
    self.ln1 = nn.LayerNorm(256)
    self.fc2 = nn.Linear(256, 256)
    self.ln2 = nn.LayerNorm(256)
    self.fc3 = nn.Linear(256, 256)
    self.ln3 = nn.LayerNorm(256)
    self.fc4 = nn.Linear(256, 10)

  def forward(self,x):
    x = self.ln1(F.relu(self.fc1(x)))
    x = self.ln2(F.relu(self.fc2(x)))
    x = self.ln3(F.relu(self.fc3(x)))
    return self.fc4(x)
  
xent = nn.CrossEntropyLoss()
model = maybe_cuda(MLP())
opt = torch.optim.Adam(model.parameters(), lr=0.001)

#@title { vertical-output: true}
NEP = 16
for ep in range(NEP):
  losses = []
  indices = torch.randperm(len(Xtr))
  for b0 in range(0, len(Xtr), 32):
    x = Variable(Xtr[indices[b0:b0+32]])
    y = Variable(Ytr[indices[b0:b0+32]])

    opt.param_groups[0]['lr'] = 0.001 * np.cos(ep / NEP * np.pi/2)  # cos decay

    opt.zero_grad()
    loss = xent(model(x), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

  corrects = []
  for i0 in range(0, len(Xte), 128):
    with torch.no_grad():
      preds = model(Xte[i0:i0+128])
      corrects.extend((preds.argmax(1) == Yte[i0:i0+128]).cpu().tolist())

  print(f'After {ep+1}ep: avg loss {np.mean(losses):.2f} ; test acc {np.mean(corrects):.1%} (={1-np.mean(corrects):.1%} error)')