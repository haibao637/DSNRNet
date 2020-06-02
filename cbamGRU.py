import torch.nn as nn
from torch.autograd import Variable
import torch


import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)


    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes*2, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))


    def forward(self, x):
        # avgout = self.sharedMLP(self.avg_pool(x))
        # maxout = self.sharedMLP(self.max_pool(x))
        # return (avgout + maxout)
        return self.sharedMLP(torch.cat([self.avg_pool(x),self.max_pool(x)],1))
class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM,self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x,y=None):
        if y is None:
            y = x
        x = self.ca(x) * x
        x = self.sa(x)*x
        return x

class CCBGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(CCBGRUCell, self).__init__()


        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.hidden_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim ,
                              out_channels= self.hidden_dim,
                              kernel_size= self.kernel_size,
                              padding=self.padding,
                              bias=self.bias),
            nn.LeakyReLU()
        )

        self.t_gate_conv = nn.Conv2d(in_channels=self.input_dim *2,
                              out_channels= self.hidden_dim*2,
                              kernel_size= self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        # self.cbam = CBAM(self.hidden_dim*2)

        self.c_gate_conv = ChannelAttention(self.hidden_dim*2) # mask

        self.s_gate_conv = SpatialAttention() # mask




        self.output_conv = nn.Conv2d(in_channels= self.hidden_dim * 2 ,
                              out_channels= self.hidden_dim,
                              kernel_size= self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, h_state):

        hidden = self.hidden_conv(input_tensor)

        combined = torch.cat([hidden,h_state],1)


        t_gate_mask = self.t_gate_conv(combined) # mask
        s_gate_mask = self.s_gate_conv(combined)
        c_gate_mask = self.c_gate_conv(combined)

        gates = F.sigmoid(t_gate_mask+s_gate_mask+c_gate_mask)

        z_gate,r_gate = torch.split(gates,self.hidden_dim,dim=1)

        new_h = F.tanh(self.output_conv(torch.cat([r_gate*h_state,hidden],1)))

        new_h = (1-z_gate)*h_state+z_gate*new_h

        return new_h,new_h




    def init_hidden(self, batch_size, tensor_size):
        height, width = tensor_size
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())


class CCBGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(CCBGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')



        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(CCBGRUCell(  input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3),input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),tensor_size=tensor_size)

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 h_state=h)
                # output_inner.append(h)

            # layer_output = torch.stack(output_inner, dim=1)
            # cur_layer_input = layer_output

            # layer_output_list.append(layer_output)
            # last_state_list.append([h])


        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list   = last_state_list[-1:]

        return h,c

    def _init_hidden(self, batch_size, tensor_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, tensor_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class CCBBGRU(nn.Module):
    # Constructor
    def __init__(self, input_dim, hidden_dim,
                 kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):

        super(CCBBGRU, self).__init__()
        self.forward_net = CCBGRU(input_dim, hidden_dim//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias,
                                    return_all_layers=return_all_layers)
        self.reverse_net = CCBGRU( input_dim, hidden_dim//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias,
                                    return_all_layers=return_all_layers)

    def forward(self, xforward, xreverse):
        """
        xforward, xreverse = B T C H W tensors.
        """

        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)

        if not self.return_all_layers:
            y_out_fwd = y_out_fwd[-1] # outputs of last CLSTM layer = B, T, C, H, W
            y_out_rev = y_out_rev[-1] # outputs of last CLSTM layer = B, T, C, H, W

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, reversed_idx, ...] # reverse temporal outputs.
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)

        return ycat