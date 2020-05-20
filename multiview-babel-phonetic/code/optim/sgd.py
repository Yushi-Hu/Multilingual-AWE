import torch.optim as optim

import utils.saver


class SGD(optim.SGD, utils.saver.Saver):

  name = "optimizer-SGD"

  def __init__(self, params, lr=0.01, momentum=0.9,
               dampening=0, weight_decay=0, nesterov=True):

    optim.SGD.__init__(self, params, lr=lr, momentum=momentum,
                       dampening=dampening, weight_decay=weight_decay,
                       nesterov=nesterov)
    utils.saver.Saver.__init__(self)
    self.converged = False
