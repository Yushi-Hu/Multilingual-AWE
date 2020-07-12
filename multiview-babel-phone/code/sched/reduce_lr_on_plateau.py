import torch.optim.lr_scheduler as lr_scheduler

import utils.saver


class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau, utils.saver.Saver):

  name = "scheduler-ReduceLROnPlateau"

  def __init__(self, optimizer, mode="min", factor=0.1, patience=10,
               min_lr=1e-8, threshold=1e-6, threshold_mode="abs", cooldown=0):

    lr_scheduler.ReduceLROnPlateau.__init__(self,
                                            optimizer=optimizer,
                                            mode=mode,
                                            factor=factor,
                                            patience=patience,
                                            threshold=threshold,
                                            threshold_mode=threshold_mode,
                                            cooldown=cooldown,
                                            min_lr=min_lr,
                                            verbose=False,
                                            eps=1e-12)
    utils.saver.Saver.__init__(self)
