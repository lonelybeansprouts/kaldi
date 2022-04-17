import time
import numpy as np
import torch
import torch.nn as nn


class FSMNKernelParallel(nn.Module):

    def __init__(self, dims, l_order, r_order, l_stride=1, r_stride=1, kernel_res=True):
        super().__init__()
        assert l_stride == r_stride == 1, f'Parallel version expected l_stride == r_stride == 1, ' \
                                          f'but get ({l_stride}, {r_stride})'
        self.filter = nn.Conv1d(in_channels=dims, out_channels=dims, kernel_size=l_order+r_order+1, stride=l_stride,
                                groups=dims, padding=0, bias=False)
        self.l_order = l_order
        self.r_order = r_order
        self.l_stride = l_stride
        self.r_stride = r_stride
        self.dims = dims
        self.register_buffer("left_padding", torch.zeros(1, self.l_order * self.l_stride, self.dims))
        self.register_buffer("right_padding", torch.zeros(1, self.r_order * self.r_stride, self.dims))
        self.kernel_res = kernel_res

    def extra_repr(self) -> str:
        return f'(l_order={self.l_order}, r_order={self.r_order}, ' \
               f'l_stride={self.l_stride}, r_stride={self.r_stride}, ' \
               f'dims={self.dims})'

    def forward(self, x):
        res = x

        x = torch.cat(
            (
                self.left_padding.expand(x.size(0), -1, -1),
                x,
                self.right_padding.expand(x.size(0), -1, -1)
            ),
            dim=1
        )
        x = x.transpose(1, 2).contiguous()  # BxTxD -> BxDxT for conv accept channel as second dimension.
        x = self.filter(x)
        y = x.transpose(1, 2).contiguous()  # BxDxT -> BxTxD
        if self.kernel_res:
            y = y + res

        return y


class DFSMNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, lo, ro,
                 skip_connection=None, activation=nn.ReLU6(True)):
        super(DFSMNLayer, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.fsmn = FSMNKernelParallel(proj_dim, lo, ro)
        self.activation = activation
        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.activation(self.hidden(x)) # input mapping
        proj = self.proj(out) # reduce dimension
        out = self.fsmn(proj) # fsmn
        if self.skip_connection == 'res':
            out += x
        return out


class DFSMN(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, lo, ro,
                 nlayer=1, skip_connection=None, activation=nn.ReLU6(True)):
        super(DFSMN, self).__init__()
        self.fsmns = nn.ModuleList([
            DFSMNLayer(
                proj_dim if i > 0 else input_dim,
                hidden_dim, proj_dim, lo, ro,
                skip_connection if i > 0 else None, activation)
            for i in range(nlayer)
        ])

    def forward(self, x):
        for fsmn in self.fsmns:
            x = fsmn(x)
        return x


class DFSMNAM(nn.Module):
    def __init__(self, input_size, hidden_dim, proj_dim, lo, ro,
                 output_size, skip='res', nlayer=15, ndnn=2,
                 activation=nn.ReLU6(True)):
        super(DFSMNAM, self).__init__()
        self.fsmn = DFSMN(input_size, hidden_dim, proj_dim, lo, ro,
                          nlayer, skip, activation)

        if ndnn > 0:
            dnns = [nn.Linear(proj_dim, hidden_dim, bias=True), activation]
            for _ in range(ndnn-1):
                dnns.extend([nn.Linear(hidden_dim, hidden_dim, bias=True), activation])
            output_in = hidden_dim
        else:
            dnns = []
            output_in = proj_dim

        self.dnn = nn.Sequential(*dnns)
        self.output = nn.Linear(output_in, output_size, bias=True)
        # self.logsoftmax = nn.LogSoftmax(2)

    def forward(self, x):
        """

        Args:
            x: BxTxD

        Returns:

        """
        out = self.fsmn(x)
        out = self.dnn(out)
        out = self.output(out)
        # out = self.logsoftmax(out)
        return out

