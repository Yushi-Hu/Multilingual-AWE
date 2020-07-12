import torch.optim.lr_scheduler as lr_scheduler

import utils.saver


class ExponentialLR(lr_scheduler.ExponentialLR, utils.saver.Saver):

  name = "scheduler-ExponentialLR"

  def __init__(self, optimizer, gamma, last_epoch=-1):

    lr_scheduler.ExponentialLR.__init__(self,
                                        optimizer=optimizer,
                                        gamma=gamma,
                                        last_epoch=last_epoch)
    utils.saver.Saver.__init__(self)
