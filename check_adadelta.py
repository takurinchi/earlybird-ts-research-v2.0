import torch
from models.mnist_lenet import create_lenet_bn, structurally_prune_lenet
from pruning.channel_manager import BNChannelMaskManager
from evaluation.structured_metrics import estimate_lenet_compact_flops, estimate_lenet_compact_parameters
from evaluation.flops import count_dense_flops
from training.optimizer_state import migrate_optimizer_state

torch.manual_seed(1)
m=create_lenet_bn()
opt=torch.optim.Adadelta(m.parameters(),lr=1.0)
x=torch.randn(4,1,32,32)
m(x).sum().backward(); opt.step(); opt.zero_grad()
masks={'features.1':torch.tensor([1,0,1,0,1,0],dtype=torch.bool),
       'features.5':torch.arange(16)%2==0}
c,slices,_=structurally_prune_lenet(m,masks,True)
newopt=torch.optim.Adadelta(c.parameters(),lr=1.0)
migrate_optimizer_state(opt,newopt,m,c,slices)
gate=BNChannelMaskManager(m,masks)
m.eval(); c.eval()
torch.testing.assert_close(m(x),c(x),rtol=1e-4,atol=1e-6)
assert estimate_lenet_compact_parameters(m,masks)==sum(p.numel() for p in c.parameters())
assert estimate_lenet_compact_flops(m,masks)==count_dense_flops(c,(1,1,32,32))
for name,p in c.named_parameters():
    assert newopt.state[p]['square_avg'].shape==p.shape
    assert newopt.state[p]['acc_delta'].shape==p.shape
    original=dict(m.named_parameters())[name]
    for key in ('square_avg','acc_delta'):
        expected=opt.state[original][key]
        for dim,idx in slices.get(name, ()):
            expected=expected.index_select(dim,idx)
        torch.testing.assert_close(newopt.state[p][key],expected)
    torch.testing.assert_close(newopt.state[p]['step'],opt.state[original]['step'])
gate.close()
print('PASS: compact forward equivalence, parameter/FLOPs estimates, Adadelta state values')

from training.builders import build_scheduler
scheduler=build_scheduler(newopt, {'lr_schedule':{'enabled':False}})
for _ in range(35):
    newopt.step(); scheduler.step()
    assert newopt.param_groups[0]['lr']==1.0
print('PASS: constant lr=1.0')
