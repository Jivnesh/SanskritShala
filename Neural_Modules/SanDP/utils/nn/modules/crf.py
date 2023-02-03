import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils.nlinalg import logsumexp, logdet
from utils.tasks import parse
from .attention import BiAAttention


class ChainCRF(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
        '''
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram


        # state weight tensor
        self.state_nn = nn.Linear(input_size, self.num_labels)
        if bigram:
            # transition weight tensor
            self.trans_nn = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter('trans_matrix', None)
        else:
            self.trans_nn = None
            self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_nn.bias, 0.)
        if self.bigram:
            nn.init.xavier_uniform_(self.trans_nn.weight)
            nn.init.constant_(self.trans_nn.bias, 0.)
        else:
            nn.init.normal_(self.trans_matrix)
        # if not self.bigram:
        #     nn.init.normal(self.trans_matrix)

    def forward(self, input, mask=None):
        '''

        Args:
            input: Tensor
                the input tensor with shape = [batch_size, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch_size, length]

        Returns: Tensor
            the energy tensor with shape = [batch_size, length, num_label, num_label]

        '''
        batch_size, length, _ = input.size()

        # compute out_s by tensor dot [batch_size, length, input_size] * [input_size, num_label]
        # thus out_s should be [batch_size, length, num_label] --> [batch_size, length, num_label, 1]
        out_s = self.state_nn(input).unsqueeze(-1)

        if self.bigram:
            # compute out_s by tensor dot: [batch_size, length, input_size] * [input_size, num_label * num_label]
            out_t = self.trans_nn(input).view(batch_size, length, self.num_labels, self.num_labels)
        else:
            out_t = self.trans_matrix
        # the output should be [batch_size, length, num_label, num_label]
        #output = out_t + out_s
        #if mask is not None:
        #    output = output * mask.unsqueeze(2).unsqueeze(3)
        return (out_s, out_t)

    def loss(self, energy, target, mask=None, length=None):
        '''

        Args:
            energy: Tensor
                the input tensor with shape = [batch_size, length, num_label, num_label]
            target: Tensor
                the tensor of target labels with shape [batch_size, length]
            mask:Tensor or None
                the mask tensor with shape = [batch_size, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''
        if length is not None:
            max_len = length.max()
            if energy.size(1) != max_len:
                target = target[:, :max_len]
        mask_transpose = None
        if mask is not None:
            energy = energy * mask.unsqueeze(2).unsqueeze(3)
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)

        batch_size, len, _, _ = energy.size()
        # shape = [length, batch_size, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch_size]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch_size, 1]

        # shape = [batch_size, num_label]
        partition = None
        if energy.is_cuda:
            # shape = [batch_size]
            batch_index = torch.arange(0, batch_size).long().cuda()
            prev_label = torch.cuda.LongTensor(batch_size).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch_size).cuda()
        else:
            # shape = [batch_size]
            batch_index = torch.arange(0, batch_size).long()
            prev_label = torch.LongTensor(batch_size).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch_size)

        for t in range(len):
            # shape = [batch_size, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :]
            else:
                # shape = [batch_size, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t].data]
            prev_label = target_transpose[t].data
        return logsumexp(partition, dim=1) - tgt_energy

    def decode(self, energy, mask=None, leading_symbolic=0):
        """

        Args:
            energy: Tensor
                the input tensor with shape = [length, batch_size, num_label, num_label]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: Tensor
            decoding results in shape [batch_size, length]

        """
        if mask is not None:
            energy = energy * mask.unsqueeze(2).unsqueeze(3)
        # Input should be provided as (batch_size, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, batch_size, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, batch_size, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = energy_transpose.size()
        if energy.is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            pi = torch.zeros([length, batch_size, num_label, 1]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size, 1).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            pi = torch.zeros([length, batch_size, num_label, 1])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size, 1).zero_()

        pi[0] = energy[:, 0, -1, leading_symbolic:-1].unsqueeze(-1)
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            x,y = torch.max(energy_transpose[t] + pi_prev, dim=1)
            pi[t] = x.unsqueeze(-1)
            pointer[t] = y
        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        back_pointer = back_pointer.squeeze(-1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]
        return back_pointer.transpose(0, 1) + leading_symbolic

class TreeCRF(nn.Module):
    '''
    Tree CRF layer.
    '''
    def __init__(self, input_size, num_labels, biaffine=True):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''
        super(TreeCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.attention = BiAAttention(input_size, input_size, num_labels, biaffine=biaffine)

    def forward(self, input_h, input_c, mask=None):
        '''

        Args:
            input_h: Tensor
                the head input tensor with shape = [batch_size, length, input_size]
            input_c: Tensor
                the child input tensor with shape = [batch_size, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch_size, length]
            lengths: Tensor or None
                the length tensor with shape = [batch_size]

        Returns: Tensor
            the energy tensor with shape = [batch_size, num_label, length, length]

        '''
        _, length, _ = input_h.size()
        # [batch_size, num_labels, length, length]
        output = self.attention(input_h, input_c, mask_d=mask, mask_e=mask)
        # set diagonal elements to -inf
        output = output + torch.diag(output.data.new(length).fill_(-np.inf))
        return output

    def loss(self, input_h, input_c, heads, arc_tags, mask=None, lengths=None):
        '''

        Args:
            input_h: Tensor
                the head input tensor with shape = [batch_size, length, input_size]
            input_c: Tensor
                the child input tensor with shape = [batch_size, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch_size, length]
            mask:Tensor or None
                the mask tensor with shape = [batch_size, length]
            lengths: tensor or list of int
                the length of each input shape = [batch_size]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''
        batch_size, length, _ = input_h.size()
        energy = self.forward(input_h, input_c, mask=mask)
        # [batch_size, num_labels, length, length]
        A = torch.exp(energy)
        # mask out invalid positions
        if mask is not None:
            A = A * mask.unsqueeze(1).unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)

        # sum along the label axis [batch_size, length, length]
        A = A.sum(dim=1)
        # get D [batch_size, 1, length]
        D = A.sum(dim=1, keepdim=True)

        # make sure L is positive-defined
        rtol = 1e-4
        atol = 1e-6
        D += D * rtol + atol

        # [batch_size, length, length]
        D = A.data.new(A.size()).zero_() + D
        # zeros out all elements except diagonal.
        D = D * torch.eye(length).type_as(D)

        # compute laplacian matrix
        # [batch_size, length, length]
        L = D - A

        # compute lengths
        if lengths is None:
            if mask is None:
                lengths = [length for _ in range(batch_size)]
            else:
                lengths = mask.data.sum(dim=1).long()

        # compute partition Z(x) [batch_size]
        z = energy.data.new(batch_size)
        for b in range(batch_size):
            Lx = L[b, 1:lengths[b], 1:lengths[b]]
            # print(torch.log(torch.eig(Lx.data)[0]))
            z[b] = logdet(Lx)

        # first create index matrix [length, batch_size]
        # index = torch.zeros(length, batch_size) + torch.arange(0, length).view(length, 1)
        index = torch.arange(0, length).view(length, 1).expand(length, batch_size)
        index = index.type_as(energy.data).long()
        batch_index = torch.arange(0, batch_size).type_as(energy.data).long()
        # compute target energy [length-1, batch_size]
        tgt_energy = energy[batch_index, arc_tags.data.t(), heads.data.t(), index][1:]
        # sum over dim=0 shape = [batch_size]
        tgt_energy = tgt_energy.sum(dim=0)

        return z - tgt_energy

class ChainCRF_with_LE(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
        '''
        super(ChainCRF_with_LE, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram

        # state weight tensor
        self.state_nn = nn.Linear(input_size, self.num_labels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_nn.bias, 0.)

    def forward(self, input, LE, mask=None):
        '''

        Args:
            input: Tensor
                the input tensor with shape = [batch_size, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch_size, length]

        Returns: Tensor
            the energy tensor with shape = [batch_size, length, num_label, num_label]

        '''
        batch_size, length, _ = input.size()

        # compute out_s by tensor dot [batch_size, length, input_size] * [input_size, num_label]
        # thus out_s should be [batch_size, length, num_label] --> [batch_size, length, num_label, 1]
        out_s = self.state_nn(input).unsqueeze(-1)
        out_t = torch.matmul(LE, torch.t(LE))
        # add column of zeros (for END token)
        col_zeros = torch.zeros((out_t.shape[0],1))
        row_zeros = torch.zeros((1, out_t.shape[1] + 1))
        if out_t.is_cuda:
            col_zeros = col_zeros.cuda()
            row_zeros = row_zeros.cuda()
        out_t = torch.cat((out_t, col_zeros), dim=1)
        # add row of zeros (for END token)
        out_t = torch.cat((out_t, row_zeros))
        # the output should be [batch_size, length, num_label, num_label]
        return (out_s, out_t)

    def loss(self, energy, target, mask=None, length=None):
        '''

        Args:
            energy: Tensor
                the input tensor with shape = [batch_size, length, num_label, num_label]
            target: Tensor
                the tensor of target labels with shape [batch_size, length]
            mask:Tensor or None
                the mask tensor with shape = [batch_size, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''
        if length is not None:
            max_len = length.max()
            if energy.size(1) != max_len:
                target = target[:, :max_len]
        mask_transpose = None
        if mask is not None:
            energy = energy * mask.unsqueeze(2).unsqueeze(3)
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)

        batch_size, len, _, _ = energy.size()
        # shape = [length, batch_size, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch_size]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch_size, 1]

        # shape = [batch_size, num_label]
        partition = None
        if energy.is_cuda:
            # shape = [batch_size]
            batch_index = torch.arange(0, batch_size).long().cuda()
            prev_label = torch.cuda.LongTensor(batch_size).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch_size).cuda()
        else:
            # shape = [batch_size]
            batch_index = torch.arange(0, batch_size).long()
            prev_label = torch.LongTensor(batch_size).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch_size)

        for t in range(len):
            # shape = [batch_size, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :]
            else:
                # shape = [batch_size, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t].data]
            prev_label = target_transpose[t].data
        return logsumexp(partition, dim=1) - tgt_energy

    def decode(self, energy, mask=None, leading_symbolic=0):
        """

        Args:
            energy: Tensor
                the input tensor with shape = [length, batch_size, num_label, num_label]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: Tensor
            decoding results in shape [batch_size, length]

        """
        if mask is not None:
            energy = energy * mask.unsqueeze(2).unsqueeze(3)
        # Input should be provided as (batch_size, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, batch_size, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, batch_size, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = energy_transpose.size()
        if energy.is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            pi = torch.zeros([length, batch_size, num_label, 1]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size, 1).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            pi = torch.zeros([length, batch_size, num_label, 1])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size, 1).zero_()

        pi[0] = energy[:, 0, -1, leading_symbolic:-1].unsqueeze(-1)
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            x,y = torch.max(energy_transpose[t] + pi_prev, dim=1)
            pi[t] = x.unsqueeze(-1)
            pointer[t] = y
        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        back_pointer = back_pointer.squeeze(-1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]
        return back_pointer.transpose(0, 1) + leading_symbolic
