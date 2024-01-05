import torch.optim.lr_scheduler as sch

class base():
    def __init__(self, *args):
        pass

    def step(self):
        pass

def get_sch(scheduler):
    if scheduler=='None':
        return base