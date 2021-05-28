import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        """
        LSTM model for air quality prediction

        :param input_size: (int) the length of the representation for a single timestamp
                           (a timestamp is a concentration of pollutant and thus is a single value in the simple case)
        :param hidden_layer_size: (int) hidden LSTM layer size
        :param output_size: (int) the output size of the linear layer that comes after the LSTM
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size  # TODO [tuning] different size (?)

        # TODO [tuning] BiLSTM (?)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size, bias=True)  # TODO try with bias=False (?)

    def forward(self, input_seq):
        lstm_out, (h_n, c_n) = self.lstm(input_seq.view(len(input_seq), -1, 1))
        # h_n.shape = [1, batch_size, self.hidden_layer_size]

        predictions = self.linear(h_n.view(len(input_seq), -1))
        # predictions.shape = [batch_size, 1]
        return predictions


class ConditionalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, aux_features_size=31, output_size=1):
        """
        Conditional LSTM model for air quality prediction
        inspired by https://github.com/philipperemy/cond_rnn
        in that settings we use the extra features that represent a timestamp in the following:
        1. encode the features to one-hot
        2. pass the one-hot through a linear layer
        3. use the hidden representation of the linear layer as the initial hidden state of the LSTM model

        :param input_size: (int) the length of the representation for a single timestamp
                           (a timestamp is a concentration of pollutant and thus is a single value in the simple case)
        :param aux_features_size: (int) the size of the one-hot encoded auxiliary vectors
        :param hidden_layer_size: (int) hidden LSTM layer size
        :param output_size: (int) the output size of the linear layer that comes after the LSTM
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size  # TODO [tuning] different size (?)

        # TODO [tuning] BiLSTM (?)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.linear_aux = nn.Linear(aux_features_size, hidden_layer_size, bias=True)  # TODO try with bias=False (?)
        self.linear = nn.Linear(hidden_layer_size, output_size, bias=True)  # TODO try with bias=False (?)

    def forward(self, input_seq, aux_data):
        hidden_state = self.linear_aux(aux_data)  # inspired by https://github.com/philipperemy/cond_rnn
        # hidden_state.shape = [batch_size, self.hidden_layer_size]

        # hx is a tuple of (hidden_state, cell_state) for the different samples, used as conditioning the LSTM
        hx = self.get_initial_hx(input_seq, hidden_state)

        # run through lstm and assign initial hidden, cell states (this is the conditioning on the aux_data)
        lstm_out, (h_n, c_n) = self.lstm(input=input_seq.view(len(input_seq), -1, 1), hx=hx)
        # h_n.shape = [1, batch_size, self.hidden_layer_size]

        predictions = self.linear(h_n.view(len(input_seq), -1))
        # predictions.shape = [batch_size, 1]
        return predictions

    def get_initial_hx(self, input_seq, hidden_state):
        """
        returns the initial (hidden, cell) states for the LSTM
        :param input_seq: (torch.tensor) with raw inputs of the time series data
        :param hidden_state: (torch.tensor) with the initial hidden states
        """
        num_directions = 2 if self.lstm.bidirectional else 1
        # hidden state
        hidden = hidden_state.view(self.lstm.num_layers * num_directions, len(hidden_state), -1)
        # cell state
        c_zeros = torch.zeros(self.lstm.num_layers * num_directions,
                              input_seq.size(0), self.lstm.hidden_size,
                              dtype=input_seq.dtype, device=input_seq.device)
        return hidden, c_zeros
