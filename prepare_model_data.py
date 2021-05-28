from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch


def sliding_windows(train_data, sequence_length, aux=None):
    """
    create data by sliding window for train set
    :param train_data: (np.array) all the original train values
    :param sequence_length: sequence length for training phase
    :param aux: (None or np.ndarray) if None there is no auxiliary features i.e., only raw time series values
                if np.ndarray contains one-hot encoded features
    :return: the train data as `x` list of sequences and `y` list of predictions.
              if aux is not None also returns the auxiliary data
    e.g. for train_data = [A,B,C,D,E,F,G] and sequence_length=5 >>
    x = [[A,B,C,D,E], [B,C,D,E,F]]
    y = [F, G]
    """
    x, x_aux, y = [], [], []

    for i in range(len(train_data) - sequence_length):
        _x = train_data[i: i + sequence_length]
        _y = train_data[i + sequence_length]
        x.append(_x)
        y.append(_y)
        if aux is not None:
            _x_aux = aux[i + sequence_length]
            x_aux.append(_x_aux)
    if aux is not None:
        return x, x_aux, y
    else:
        return x, y


def get_auxiliary_features(data, first_ts, dayofweek: bool, hourofday: bool):
    """
    list of tuples with additional auxiliary features for every timestamp in `data`

    :param data: (np.ndarray) 1d array of air quality concentrations data for a single location and pollutant
                 is assumed to be continuous (i.e. no missing timestamp, every timestamp is +1hr of the previous sample)
    :param first_ts: (pd.Timestamp) first time of a measurement in the time series `data`
                     e.g., Timestamp('2017-01-01 14:00:00')
    :param dayofweek: (bool) whether to encode dayofweek as a feature
    :param hourofday: (bool) whether to encode hourofday as a feature
    :return: list of auxiliary features for every timestamp, with len == len(data),
             e.g., [(6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23)]
             6 is dayofweek and 14-23 are the hours of day
    """
    aux_features_lists = list()
    if dayofweek:
        dayofweek_list = list()
        for i in range(len(data)):
            cur_ts = first_ts + pd.Timedelta(f'{i}h')
            dayofweek_list.append(cur_ts.dayofweek)
        aux_features_lists.append(dayofweek_list)

    if hourofday:
        hourofday_list = list()
        for i in range(len(data)):
            cur_ts = first_ts + pd.Timedelta(f'{i}h')
            hourofday_list.append(cur_ts.hour)
        aux_features_lists.append(hourofday_list)

    return list(zip(*aux_features_lists))


def aux_features_to_onehot(aux_features, categories):
    """
    get the one-hot encoding of the auxiliary features returned by get_auxiliary_features()
    :param aux_features: (list) result of get_auxiliary_features()
    :param categories: categories to pass for OneHotEncoder (possible values for each feature by its order)
    :return: np.ndarray of the one-hot encoded auxiliary features
    """
    enc = OneHotEncoder(categories=categories)
    return enc.fit_transform(aux_features).toarray()


def create_data_for_train_and_test(data, test_size, sequence_length, batch_size):
    """
    transform the time series values in `data` to a (train, test) split and convert train to pytorch dataset.
    each sample (x, y) is created by windowing such that x is of length `sequence_length` and `y` is a prediction of the
    next time step

    :param data: (np.ndarray) 1d array of air quality concentrations data for a single location and pollutant
    :param test_size: (int) the amount of the last samples (from the end of the array named `data`) that used as test
    :param sequence_length: (int) length of sequence that is used during training phase
    :param batch_size: (int) batch size for train phase

    :return: 1. train_dataloader - shuffled pytorch DataLoader for train data
             2. test_inputs - sequence of the last items from the train set that are used to predict the first value
                from test set (len(test_inputs) = `sequence_length`)
             3. test_y - unscaled (original) test set
             4. scaler - the scaler used for scaling data values
    """
    # define scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))  # TODO [feature engineering] another range like (0,1)

    # slice the train data
    train_data = data[:-test_size]
    train_data = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(-1)  # fit and transform data

    # `test_inputs` are used to predict the first value in `test_y` (see eval())
    test_y = data[-test_size:]
    test_inputs = list(train_data[-sequence_length:])

    # create the sliding window train data
    train_x, train_y = sliding_windows(train_data=train_data, sequence_length=sequence_length)

    # transform to torch tensor and push data to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x = torch.FloatTensor(train_x).to(device)
    train_y = torch.FloatTensor(train_y).to(device)

    # create dataset and dataloader
    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_inputs, test_y, scaler


def create_data_for_train_and_test_extra_features(data, first_ts, test_size, sequence_length, batch_size):
    """
    transform the time series values in `data` to a (train, test) split and convert train to pytorch dataset.
    each sample (x, y) is created by windowing such that x is of length `sequence_length` and `y` is a prediction of the
    next time step

    this settings also include the use of extra features besides the original time series values

    :param data: (np.ndarray) 1d array of air quality concentrations data for a single location and pollutant
    :param first_ts: (pd.Timestamp) first time of a measurement in the time series `data`
                     e.g., Timestamp('2017-01-01 14:00:00')
    :param test_size: (int) the amount of the last samples (from the end of the array named `data`) that used as test
    :param sequence_length: (int) length of sequence that is used during training phase
    :param batch_size: (int) batch size for train phase

    :return: 1. train_dataloader - shuffled pytorch DataLoader for train data
             2. test_inputs - sequence of the last items from the train set that are used to predict the first value
                from test set (len(test_inputs) = `sequence_length`)
             3. test_y - unscaled (original) test set
             4. scaler - the scaler used for scaling data values
    """
    # here goes the extra data that should be added (besides the raw time series data)
    auxiliary_features = get_auxiliary_features(data=data, first_ts=first_ts, dayofweek=True, hourofday=False)
    aux_data_onehot = aux_features_to_onehot(aux_features=auxiliary_features,
                                             categories=[list(range(7)), list(range(24))])

    # define scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))  # TODO [feature engineering] another range like (0,1)

    # slice the train data
    train_data = data[:-test_size]
    train_aux_data = aux_data_onehot[:-test_size]
    train_data = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(-1)  # fit and transform data

    # `test_inputs` are used to predict the first value in `test_y` (see eval())
    test_y = data[-test_size:]
    test_y_aux_data = aux_data_onehot[-test_size:]
    test_inputs = list(train_data[-sequence_length:])

    # create the sliding window train data
    train_x, train_x_aux, train_y = sliding_windows(train_data=train_data, sequence_length=sequence_length,
                                                    aux=train_aux_data)

    # transform to torch tensor and push data to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x = torch.FloatTensor(train_x).to(device)
    train_aux = torch.FloatTensor(train_x_aux).to(device)
    train_y = torch.FloatTensor(train_y).to(device)

    # create dataset and dataloader
    train_dataset = TensorDataset(train_x, train_aux, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_inputs, test_y, test_y_aux_data, scaler
