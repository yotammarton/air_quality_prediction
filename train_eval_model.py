import preprocess_raw_data
from prepare_model_data import create_data_for_train_and_test, create_data_for_train_and_test_extra_features
from model import LSTM, ConditionalLSTM
from utils import smape, split_array_to_size
from plot_metrics import plot_smape
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import json


def train(model, train_dataloader, optimizer, loss_function, with_aux_data=False):
    """
    one epoch train of the model
    :param model: (nn.Module) model for predictions
    :param train_dataloader: (torch.utils.data.DataLoader) shuffled pytorch DataLoader for train data
    :param optimizer: (orch.optim.*) pytorch optimizer
    :param loss_function: (nn.*) some loss function, e.g. nn.MSELoss
    :param with_aux_data: (bool) whether to train with auxiliary data or not (extra features)
    :return: list of batch losses for train data
    """
    train_epoch_losses = list()
    model.train()
    for i_batch, batch_data in enumerate(train_dataloader):
        optimizer.zero_grad()
        if with_aux_data:
            (x, aux, y) = batch_data
            output = model(x, aux)
        else:
            (x, y) = batch_data
            output = model(x)
        loss = loss_function(output, y.view(len(x), -1))
        loss.backward()
        train_epoch_losses.append(float(loss))
        optimizer.step()
    return train_epoch_losses


def eval(model, sequence_length, test_inputs, test_y, scaler, device, aux_data=None):
    """
    calculate the mse, mae, smape metrics for the true unscaled values in `test_y`
     and the predicted values produced by the model.

    :param model: (nn.Module) model for predictions
    :param sequence_length: (int) length of sequence that is used during training phase
    :param test_inputs: (list) sequence of the last items from the train set that are used to predict the first value
                        from test set
    :param test_y: (np.ndarray) unscaled (original) test set
    :param scaler: (sklearn.preprocessing) the scaler used for scaling data values
    :param device: (str) 'cuda' or 'cpu'
    :param aux_data: (None or np.ndarray) if None there is no auxiliary data.
                     otherwise, it is the one-hot encoded aux data
    :return: mse, mae, smape of predictions compared to `test_y`
    """
    model.eval()
    test_inputs_copy = list(test_inputs)  # copy so we will not overwrite
    with torch.no_grad():
        for i in range(len(test_y)):
            # get the current raw time series values for the prediction of the next value
            seq = torch.FloatTensor(test_inputs_copy[-sequence_length:]).view(1, -1).to(device)

            if aux_data is not None:  # with auxiliary data
                # get the conditioned auxiliary data for the prediction
                cur_aux = torch.FloatTensor(aux_data[i]).view(1, -1).to(device)
                test_inputs_copy.append(model(seq, cur_aux).item())
            else:  # only raw time series values
                test_inputs_copy.append(model(seq).item())

        # get the corresponding predictions for `test_y`
        scaled_predicted_test_y = np.array(test_inputs_copy[-len(test_y):])

        # inverse predictions scale to go back to original values
        unscaled_predictions = scaler.inverse_transform(scaled_predicted_test_y.reshape(-1, 1)).reshape(-1)

        # compare the unscaled predictions to `test_y`
        mse_score = mean_squared_error(y_true=test_y, y_pred=unscaled_predictions)
        mae_score = mean_absolute_error(y_true=test_y, y_pred=unscaled_predictions)
        smape_score = smape(A=test_y, F=unscaled_predictions)

        return mse_score, mae_score, smape_score


# TODO [tuning] choose epochs amount or add early-stopping mechanism (?)
def run_baseline(data_dict, cities_stations_pairs, test_size=12, sequence_length=8, max_epochs=25, verbose=False):
    """
    train and evaluate a model for every time-series
    in this settings the model learns from samples of:
        {X = [measurements 1,2,...,8], Y = measurement 9}
    the whole time-series is used except the last `test_size` samples that are to be predicted

    :param data_dict: (dict) return value of func `preprocess_raw_data.load_raw_data_to_dict`
    :param cities_stations_pairs: (list)
    :param test_size: (int) the amount of samples from the end of every time-series that are used as test set
    :param sequence_length: (int) length of sequence that is used during training phase
    :param max_epochs: (int)
    :param verbose: (bool) True will plot metrics every epoch

    :return: dict with best metrics for every time-series
    """
    chosen_models_metrics = dict()
    for city, station in cities_stations_pairs:
        for (sid, pollutant, first_measurement_datetime), values in data_dict[city][station].items():
            print(f'----------------------- {station} {sid} {pollutant} -----------------------')
            if len(set(values[-test_size:])) == 1:
                print(f'[WARNING] test data contains only a single value: {set(values[-test_size:])}\n'
                      f'trimming the zeroes from the end of the data.. '
                      f'(removed {len(values) - np.max(np.argwhere(values)) - 1})')
                values = values[: np.max(np.argwhere(values)) + 1]  # remove all the zeroes from the end of time series

            best_mse = np.inf  # per model
            # get the data
            train_dataloader, test_inputs, test_y, scaler = \
                create_data_for_train_and_test(data=values, test_size=test_size,
                                               sequence_length=sequence_length, batch_size=32)
            # define the model
            model = LSTM()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # define loss and optimizer
            loss_function = nn.MSELoss()  # TODO [optimization] another loss function (?)
            optimizer = torch.optim.Adam(model.parameters())  # TODO [tuning] parameters of optimizer (?)

            # train and evaluate `max_epochs` times
            for epoch in range(max_epochs):
                # train model
                train_epoch_losses = train(model=model, train_dataloader=train_dataloader,
                                           optimizer=optimizer, loss_function=loss_function)

                # eval model
                mse_score, mae_score, smape_score = eval(model=model, sequence_length=sequence_length,
                                                         test_inputs=test_inputs, test_y=test_y,
                                                         scaler=scaler, device=device)

                # collect best metrics on the test data (just logging, not saving the models)
                if mse_score < best_mse:
                    best_mse = mse_score
                    chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}'] = \
                        dict(mse_score=round(mse_score, 2),
                             mae_score=round(mae_score, 2),
                             smape_score=round(smape_score, 2))

                if verbose:
                    print(f"epoch {epoch + 1}/{max_epochs} mean train loss:"
                          f" {round(float(np.mean(train_epoch_losses)), 6)}")
                    print(f"[MSE score]: {round(mse_score, 2)},\t\t "
                          f"[MAE score]: {round(mae_score, 2)},\t\t "
                          f"[SMAPE score]: {round(smape_score, 2)}")

            print(f"best test metrics: {chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}']}")

    return chosen_models_metrics


# TODO [tuning] choose epochs amount or add early-stopping mechanism (?)
def run_experiment1(data_dict, cities_stations_pairs, n_parts=3, test_size=12, sequence_length=8,
                    max_epochs=25, verbose=False):
    """
    train and evaluate a model for every time-series
    in this settings the model learns from samples of:
        {X = [measurements 1,2,...,8], Y = measurement 9}
    the original time-series is divided into `n_parts`
    each part is in size ~(len(time series) / n_parts
    model is trained and tested on every such part

    :param data_dict: (dict) return value of func `preprocess_raw_data.load_raw_data_to_dict`
    :param cities_stations_pairs: (list)
    :param n_parts: (int) amount of parts to divide the original time series to
    :param test_size: (int) the amount of samples from the end of every time-series that are used as test set
    :param sequence_length: (int) length of sequence that is used during training phase
    :param max_epochs: (int)
    :param verbose: (bool) True will plot metrics every epoch

    :return: dict with best metrics for every time-series
    """
    chosen_models_metrics = dict()
    for city, station in cities_stations_pairs:
        for (sid, pollutant, first_measurement_datetime), values in data_dict[city][station].items():
            for i_part, part_values in enumerate(np.array_split(values, n_parts)):
                print(f'----------------------- {station} {sid} {pollutant}, {i_part} -----------------------')
                if len(set(part_values[-test_size:])) == 1:
                    try:  # TODO arrange
                        print(f'[WARNING] test data contains only a single value: {set(part_values[-test_size:])}\n'
                              f'trimming the zeroes from the end of the data.. '
                              f'(removed {len(part_values) - np.max(np.argwhere(part_values)) - 1})')
                        # remove all the zeroes from the end of time series
                        part_values = part_values[: np.max(np.argwhere(part_values)) + 1]

                        # make sure data of at least 2 weeks (14 days) is available for training
                        if len(part_values) < 14 * 24 + test_size:
                            print(f'[WARNING] part {i_part} size is too small ({len(part_values)}), skipping this part')
                            continue
                    except:
                        continue

                best_mse = np.inf  # per model
                # get the data
                train_dataloader, test_inputs, test_y, scaler = \
                    create_data_for_train_and_test(data=part_values, test_size=test_size,
                                                   sequence_length=sequence_length, batch_size=32)
                # define the model
                model = LSTM()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # define loss and optimizer
                loss_function = nn.MSELoss()  # TODO [optimization] another loss function (?)
                optimizer = torch.optim.Adam(model.parameters())  # TODO [tuning] parameters of optimizer (?)

                # train and evaluate `max_epochs` times
                for epoch in range(max_epochs):
                    # train model
                    train_epoch_losses = train(model=model, train_dataloader=train_dataloader,
                                               optimizer=optimizer, loss_function=loss_function)

                    # eval model
                    mse_score, mae_score, smape_score = eval(model=model, sequence_length=sequence_length,
                                                             test_inputs=test_inputs, test_y=test_y,
                                                             scaler=scaler, device=device)

                    # collect best metrics on the test data (just logging, not saving the models)
                    if mse_score < best_mse:
                        best_mse = mse_score
                        chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}, {i_part}'] = \
                            dict(mse_score=round(mse_score, 2),
                                 mae_score=round(mae_score, 2),
                                 smape_score=round(smape_score, 2))

                    if verbose:
                        print(f"epoch {epoch + 1}/{max_epochs} mean train loss:"
                              f" {round(float(np.mean(train_epoch_losses)), 6)}")
                        print(f"[MSE score]: {round(mse_score, 2)},\t\t "
                              f"[MAE score]: {round(mae_score, 2)},\t\t "
                              f"[SMAPE score]: {round(smape_score, 2)}")

                print(f"best test metrics: {chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}, {i_part}']}")

    return chosen_models_metrics


# TODO [tuning] choose epochs amount or add early-stopping mechanism (?)
def run_experiment2(data_dict, cities_stations_pairs, train_size=14 * 24, test_size=12, sequence_length=8,
                    max_epochs=25, verbose=False):
    """
    train and evaluate a model for every time-series
    in this settings the model learns from samples of:
        {X = [measurements 1,2,...,8], Y = measurement 9}
    the original time-series is divided into chunks of size (`train_size` + `test_size`)
    model is trained and tested on every such chunk

    :param data_dict: (dict) return value of func `preprocess_raw_data.load_raw_data_to_dict`
    :param cities_stations_pairs: (list)
    :param train_size: (int) size of training set for every model (every sample represents 1 sample, i.e., 1 hour read)
    :param test_size: (int) the amount of samples from the end of every time-series that are used as test set
    :param sequence_length: (int) length of sequence that is used during training phase
    :param max_epochs: (int)
    :param verbose: (bool) True will plot metrics every epoch

    :return: dict with best metrics for every time-series
    """
    chosen_models_metrics = dict()
    for city, station in cities_stations_pairs:
        for (sid, pollutant, first_measurement_datetime), values in data_dict[city][station].items():
            parts = split_array_to_size(arr=values, size=train_size + test_size, drop_last=True)
            for i_part, part_values in enumerate(parts):
                print(f'----------------------- {station} {sid} {pollutant}, {i_part} -----------------------')
                if len(set(part_values[-test_size:])) == 1:
                    print(f'[WARNING] test data contains only a single value: {set(part_values[-test_size:])} '
                          f'continuing ...')
                    continue

                best_mse = np.inf  # per model
                # get the data
                train_dataloader, test_inputs, test_y, scaler = \
                    create_data_for_train_and_test(data=part_values, test_size=test_size,
                                                   sequence_length=sequence_length, batch_size=32)
                # define the model
                model = LSTM()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # define loss and optimizer
                loss_function = nn.MSELoss()  # TODO [optimization] another loss function (?)
                optimizer = torch.optim.Adam(model.parameters())  # TODO [tuning] parameters of optimizer (?)

                # train and evaluate `max_epochs` times
                for epoch in range(max_epochs):
                    # train model
                    train_epoch_losses = train(model=model, train_dataloader=train_dataloader,
                                               optimizer=optimizer, loss_function=loss_function)

                    # eval model
                    mse_score, mae_score, smape_score = eval(model=model, sequence_length=sequence_length,
                                                             test_inputs=test_inputs, test_y=test_y,
                                                             scaler=scaler, device=device)

                    # collect best metrics on the test data (just logging, not saving the models)
                    if mse_score < best_mse:
                        best_mse = mse_score
                        chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}, {i_part}'] = \
                            dict(mse_score=round(mse_score, 2),
                                 mae_score=round(mae_score, 2),
                                 smape_score=round(smape_score, 2))

                    if verbose:
                        print(f"epoch {epoch + 1}/{max_epochs} mean train loss:"
                              f" {round(float(np.mean(train_epoch_losses)), 6)}")
                        print(f"[MSE score]: {round(mse_score, 2)},\t\t "
                              f"[MAE score]: {round(mae_score, 2)},\t\t "
                              f"[SMAPE score]: {round(smape_score, 2)}")

                print(f"best test metrics: {chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}, {i_part}']}")

    return chosen_models_metrics


# TODO [tuning] choose epochs amount or add early-stopping mechanism (?)
def run_experiment3(data_dict, cities_stations_pairs, train_size=14 * 24, test_size=12, sequence_length=8,
                    max_epochs=25, verbose=False):
    """
    train and evaluate a model for every time-series
    in this settings the model learns from samples of:
        {X = [measurements 1,2,...,8], Y = measurement 9}
    the original time-series is divided into chunks of size (train_size + test_size)
    model is trained and tested on every such chunk

    this settings also include the use of extra features besides the original time series values
    (such as dayofweek, hour)

    :param data_dict: (dict) return value of func `preprocess_raw_data.load_raw_data_to_dict`
    :param cities_stations_pairs: (list)
    :param train_size: (int) size of training set for every model (every sample represents 1 sample, i.e., 1 hour read)
    :param test_size: (int) the amount of samples from the end of every time-series that are used as test set
    :param sequence_length: (int) length of sequence that is used during training phase
    :param max_epochs: (int)
    :param verbose: (bool) True will plot metrics every epoch

    :return: dict with best metrics for every time-series
    """
    chosen_models_metrics = dict()
    for city, station in cities_stations_pairs:
        for (sid, pollutant, first_measurement_datetime), values in data_dict[city][station].items():
            parts = split_array_to_size(arr=values, size=train_size + test_size, drop_last=True)
            for i_part, part_values in enumerate(parts):
                print(f'----------------------- {station} {sid} {pollutant}, {i_part} -----------------------')
                if len(set(part_values[-test_size:])) == 1:
                    print(f'[WARNING] test data contains only a single value: {set(part_values[-test_size:])} '
                          f'continuing ...')
                    continue

                first_ts = pd.to_datetime(first_measurement_datetime, format="%Y-%m-%d %H-%M-%S")
                delta = pd.Timedelta(f'{i_part * (train_size + test_size)}h')
                best_mse = np.inf  # per model
                # get the data
                train_dataloader, test_inputs, test_y, test_y_aux_data, scaler = \
                    create_data_for_train_and_test_extra_features(data=part_values,
                                                                  first_ts=first_ts + delta,
                                                                  test_size=test_size,
                                                                  sequence_length=sequence_length, batch_size=32)
                # define the model
                model = ConditionalLSTM()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # define loss and optimizer
                loss_function = nn.MSELoss()  # TODO [optimization] another loss function (?)
                optimizer = torch.optim.Adam(model.parameters())  # TODO [tuning] parameters of optimizer (?)

                # train and evaluate `max_epochs` times
                for epoch in range(max_epochs):
                    # train model
                    train_epoch_losses = train(model=model, train_dataloader=train_dataloader,
                                               optimizer=optimizer, loss_function=loss_function,
                                               with_aux_data=True)

                    # eval model
                    mse_score, mae_score, smape_score = eval(model=model, sequence_length=sequence_length,
                                                             test_inputs=test_inputs, test_y=test_y,
                                                             scaler=scaler, device=device, aux_data=test_y_aux_data)

                    # collect best metrics on the test data (just logging, not saving the models)
                    if mse_score < best_mse:
                        best_mse = mse_score
                        chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}, {i_part}'] = \
                            dict(mse_score=round(mse_score, 2),
                                 mae_score=round(mae_score, 2),
                                 smape_score=round(smape_score, 2))

                    if verbose:
                        print(f"epoch {epoch + 1}/{max_epochs} mean train loss:"
                              f" {round(float(np.mean(train_epoch_losses)), 6)}")
                        print(f"[MSE score]: {round(mse_score, 2)},\t\t "
                              f"[MAE score]: {round(mae_score, 2)},\t\t "
                              f"[SMAPE score]: {round(smape_score, 2)}")

                print(f"best test metrics: {chosen_models_metrics[f'{city}, {station}, {sid}, {pollutant}, {i_part}']}")

    return chosen_models_metrics


if __name__ == '__main__':
    path = '[original] kdd_cup_2018_dataset_with_missing_values_as_zeros.txt'
    data_dict = preprocess_raw_data.load_raw_data_to_dict(path)

    """BASELINE"""
    # results1 = run_baseline(data_dict=data_dict,
    #                         cities_stations_pairs=[('Beijing', station) for station in data_dict['Beijing'].keys()] +
    #                                               [('London', station) for station in data_dict['London'].keys()],
    #                         test_size=12, sequence_length=8, verbose=False)
    # i = 0
    # with open(f'results{i + 1}.json', 'w') as f:
    #     json.dump(results1, f)
    #
    # # load dict
    # with open(f'results{i + 1}.json', 'r') as f:
    #     data = json.load(f)
    #
    # # plot results
    # plot_smape(data, filename=f'metrics{i + 1}.html')
    #
    # """EXPERIMENT 1"""
    # results2 = run_experiment1(data_dict=data_dict,
    #                            cities_stations_pairs=[('Beijing', station) for station in data_dict['Beijing'].keys()] +
    #                                                  [('London', station) for station in data_dict['London'].keys()],
    #                            n_parts=3, test_size=12, sequence_length=8, verbose=False)
    # i = 1
    # with open(f'results{i + 1}.json', 'w') as f:
    #     json.dump(results2, f)
    #
    # # load dict
    # with open(f'results{i + 1}.json', 'r') as f:
    #     data = json.load(f)
    #
    # # plot results
    # plot_smape(data, filename=f'metrics{i + 1}.html')
    #
    # """EXPERIMENT 2"""
    # results3 = run_experiment2(data_dict=data_dict,
    #                            cities_stations_pairs=[('Beijing', station) for station in data_dict['Beijing'].keys()] +
    #                                                  [('London', station) for station in data_dict['London'].keys()],
    #                            train_size=14 * 24,  # 14 days * 24 hours
    #                            test_size=12, sequence_length=8, verbose=False)
    # i = 2
    # with open(f'results{i + 1}.json', 'w') as f:
    #     json.dump(results3, f)
    #
    # # load dict
    # with open(f'results{i + 1}.json', 'r') as f:
    #     data = json.load(f)
    #
    # # plot results
    # plot_smape(data, filename=f'metrics{i + 1}.html')

    """EXPERIMENT 3"""
    results6 = run_experiment2(data_dict=data_dict,
                               cities_stations_pairs=[('Beijing', station) for station in data_dict['Beijing'].keys()] +
                                                     [('London', station) for station in data_dict['London'].keys()],
                               train_size=14 * 24,  # 14 days * 24 hours
                               test_size=12, sequence_length=8, verbose=False)
    i = 5
    with open(f'results{i + 1}.json', 'w') as f:
        json.dump(results6, f)

    # load dict
    with open(f'results{i + 1}.json', 'r') as f:
        data = json.load(f)

    # plot results
    plot_smape(data, filename=f'metrics{i + 1}.html')

    # for i, results in enumerate([results1, results2, results3, results4]):
    #     # save dict
    #     with open(f'results{i + 1}.json', 'w') as f:
    #         json.dump(results, f)
    #
    #     # load dict
    #     with open(f'results{i + 1}.json', 'r') as f:
    #         data = json.load(f)
    #
    #     # plot results
    #     plot_smape(data, filename=f'metrics{i + 1}.html')
