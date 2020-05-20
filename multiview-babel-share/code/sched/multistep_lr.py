import torch.optim.lr_scheduler as lr_scheduler

import utils.saver


class MultiStepLR(lr_scheduler.MultiStepLR, utils.saver.Saver):

  name = "scheduler-MultiStepLR"

  def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):

    lr_scheduler.MultiStepLR.__init__(self,
                                      optimizer=optimizer,
                                      milestones=milestones,
                                      gamma=gamma,
                                      last_epoch=last_epoch)
    utils.saver.Saver.__init__(self)
