import logging as log
import torch.optim.lr_scheduler as lr_scheduler

import utils.saver


class RevertOnPlateau(lr_scheduler.ReduceLROnPlateau, utils.saver.Saver):

  name = "scheduler-RevertOnPlateau"

  def __init__(self, network, optimizer,
               mode="min", factor=0.1, patience=10, min_lr=1e-8,
               threshold=1e-6, threshold_mode="abs", cooldown=0):

    self.network = network

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

  def step(self, metrics, global_step):

    current = float(metrics)
    best_so_far = self.is_better(current, self.best)

    if best_so_far:
      self.best = current
      self.num_bad_epochs = 0
    else:
      self.num_bad_epochs += 1

    if self.in_cooldown:
      self.cooldown_counter -= 1
      self.num_bad_epochs = 0

    if self.num_bad_epochs > self.patience:

      self.network.load("best")  # reset network to last best
      self.optimizer.load("best")  # reset optimizer to last best

      for i, param_group in enumerate(self.optimizer.param_groups):
        lr = float(param_group["lr"])
        if lr > self.min_lrs[i] + self.eps:
          new_lr = max(lr * self.factor, self.min_lrs[i])
          log.info(f"Reducing group {i} lr from {lr} to {new_lr}.")
          param_group["lr"] = new_lr
        else:
          log.info("Training converged.")
          self.optimizer.converged = True

      self.optimizer.save(global_step, best=True)

      self.num_bad_epochs = 0

    return best_so_far
