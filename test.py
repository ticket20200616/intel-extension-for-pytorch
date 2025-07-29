import os
import torch

def log(msg):
    print(f'[Pyt]: {msg}')

if os.getenv('USE_VIRTD'):
    log('init virtd backend')
    import virtd
    virtd.init()
    dev = torch.device('virtd')
else:
    dev = torch.device('cpu')

log('define a')
a = torch.tensor([1.0, 2.0, 3.0], device=dev)

log('define b')
b = torch.tensor([4.0, 5.0, 6.0], device=dev)

log('perform a+b')
c = a + b

log('print result')
# log(c) # must implement `view` operator, then can be printed

log(c.cpu()) # move to CPU for printing
