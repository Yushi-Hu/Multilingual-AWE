import torch

import utils.saver


class Linear(torch.nn.Module, utils.saver.Saver):

  def __init__(self, in_features, out_features, bias=True):
    super(Linear, self).__init__()

    self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, x):

    x = x.to(self.device)
    # x = x.contiguous()
    # dims = list(x.shape)
    """
    if len(dims) > 2:
      d = dims.pop()
      n = torch.prod(torch.tensor(dims, dtype=torch.long))
      return self.linear(x.view(n, d)).view(*dims, -1)
    else:
    """
    return self.linear(x)
