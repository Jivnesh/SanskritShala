from ..nn import Embedding
from ..nn import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sequence_Tagger(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, use_pos, use_char, pos_dim, num_pos,
                 num_filters, kernel_size, rnn_mode, hidden_size, num_layers, tag_space, num_tags,
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 initializer=None):
        super(Sequence_Tagger, self).__init__()
        self.rnn_encoder = BiRecurrentConv_Encoder(word_dim, num_words, char_dim, num_chars, use_pos, use_char,
                                                   pos_dim, num_pos, num_filters,
                                                   kernel_size, rnn_mode, hidden_size,
                                                   num_layers, embedd_word=embedd_word,
                                                   embedd_char=embedd_char, embedd_pos=embedd_pos,
                                                   p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)
        self.sequence_tagger_decoder = Tagger_Decoder(hidden_size, tag_space, num_tags, p_out, initializer)

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        encoder_output, hn, mask, length = self.rnn_encoder(input_word, input_char, input_pos, mask, length, hx)
        out_counter = self.sequence_tagger_decoder(encoder_output, mask)
        return out_counter, mask, length

    def loss(self, input, target, mask=None, length=None):
        loss_ = self.sequence_tagger_decoder.loss(input, target, mask, length)
        return loss_

    def decode(self, input, mask=None, length=None, leading_symbolic=0):
        out_pred = self.sequence_tagger_decoder.decode(input, mask, leading_symbolic)
        return out_pred

class Tagger_Decoder(nn.Module):
    def __init__(self, hidden_size, tag_space, num_tags, p_out, initializer):
        super(Tagger_Decoder, self).__init__()
        self.criterion_obj = nn.CrossEntropyLoss()
        self.tag_space = tag_space
        self.num_tags = num_tags
        self.p_out = p_out
        self.initializer = initializer
        self.dropout_out = nn.Dropout(p_out)
        self.out_dim = 2 * hidden_size
        self.num_tags = num_tags
        self.fc_1 = nn.Linear(self.out_dim, tag_space)
        self.fc_2 = nn.Linear(tag_space, tag_space//2)
        self.fc_3 = nn.Linear(tag_space//2, num_tags)
        self.reset_parameters()

    def reset_parameters(self):
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if parameter.dim() == 1:
                parameter.data.zero_()
            else:
                self.initializer(parameter.data)

    def forward(self, input, mask):
        # input from rnn [batch_size, length, hidden_size]
        # [batch_size, length, tag_space]
        output = self.dropout_out(F.elu(self.fc_1(input)))
        #output = self.fc_2(output)
        output = self.dropout_out(F.elu(self.fc_2(output)))
        output = self.fc_3(output)
        return output

    def loss(self, input, target, mask=None, length=None):
        if length is not None:
            max_len = length.max()
            if target.size(1) != max_len:
                target = target[:, :max_len]
        input = input.view(-1, self.num_tags)
        target = target.contiguous().view(-1)
        loss_ = self.criterion_obj(input, target)
        return loss_

    def decode(self, input, mask=None, leading_symbolic=0):
        if mask is not None:
            input = input * mask.unsqueeze(2)
        # remove the first #symbolic rows and columns.
        # now the shape of the input is [n_time_steps, batch_size, t] where t = num_labels - #symbolic.
        input = input[:, :, :-leading_symbolic]
        preds = torch.argmax(input, -1)
        return preds

class BiRecurrentConv_Encoder(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, use_pos, use_char, pos_dim, num_pos, num_filters,
                 kernel_size, rnn_mode, hidden_size, num_layers, embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), initializer=None):
        super(BiRecurrentConv_Encoder, self).__init__()
        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char) if use_char else None
        self.pos_embedd = Embedding(num_pos, pos_dim, init_embedding=embedd_pos) if use_pos else None
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if use_char else None
        # dropout word
        self.dropout_in = nn.Dropout2d(p_in)
        # standard dropout
        self.dropout_out = nn.Dropout2d(p_out)
        self.dropout_rnn_in = nn.Dropout(p_rnn[0])
        self.use_pos = use_pos
        self.use_char = use_char
        self.rnn_mode = rnn_mode
        self.dim_enc = word_dim
        if use_pos:
            self.dim_enc += pos_dim
        if use_char:
            self.dim_enc += num_filters

        if rnn_mode == 'RNN':
            RNN = nn.RNN
            drop_p_rnn = p_rnn[1]
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
            drop_p_rnn = p_rnn[1]
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
            drop_p_rnn = p_rnn[1]
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        self.rnn = RNN(self.dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                       dropout=drop_p_rnn)
        self.initializer = initializer
        self.reset_parameters()

    def reset_parameters(self):
        if self.initializer is None:
            return

        for name, parameter in self.named_parameters():
            if name.find('embedd') == -1:
                if parameter.dim() == 1:
                    parameter.data.zero_()
                else:
                    self.initializer(parameter.data)

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()

        # [batch_size, length, word_dim]
        word = self.word_embedd(input_word)
        # apply dropout on input
        word = self.dropout_in(word)

        input = word
        if self.use_char:
            # [batch_size, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.conv1d(char).max(dim=2)
            # reshape to [batch_size, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout on input
            char = self.dropout_in(char)
            # concatenate word and char [batch_size, length, word_dim+char_filter]
            input = torch.cat([input, char], dim=2)

        if self.use_pos:
            # [batch_size, length, pos_dim]
            pos = self.pos_embedd(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            input = torch.cat([input, pos], dim=2)

        # apply dropout rnn input
        input = self.dropout_rnn_in(input)
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
            self.rnn.flatten_parameters()
            seq_output, hn = self.rnn(seq_input, hx=hx)
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch_size, length, hidden_size]
            self.rnn.flatten_parameters()
            output, hn = self.rnn(input, hx=hx)
        # apply dropout for the output of rnn
        output = self.dropout_out(output)
        return output, hn, mask, length