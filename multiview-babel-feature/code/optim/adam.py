import torch.optim as optim

import utils.saver


class Adam(optim.Adam, utils.saver.Saver):

  name = "optimizer-Adam"

  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
               weight_decay=0, amsgrad=False):

    optim.Adam.__init__(self, params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
    utils.saver.Saver.__init__(self)
    self.converged = False
