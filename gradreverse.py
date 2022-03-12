from torch.autograd import Function

class ReverseLayerF(Function):

    def forward(self, x, args):
        self.lambd = args.lambd
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, args):
    return ReverseLayerF().forward(x, args)