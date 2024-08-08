import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
import torch

d = 2  # dimension of the input features
widths = [64, 64]  # hidden channel dimensions
f = net.NN(lay.singleLayer(d, widths[0], act=act.tanhActivation()),
           lay.singleLayer(widths[0], widths[1], act=act.tanhActivation()),
           lay.singleLayer(widths[1], 1, act=act.tanhActivation())
           )

nex = 100  # number of examples
x = torch.randn(nex, d)
fx, dfx, lapfd2x = f(x, do_gradient=True, do_Laplacian=True, forward_mode=True)
print(fx.shape, dfx.shape, lapfd2x.shape)
