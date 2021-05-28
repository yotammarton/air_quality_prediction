import numpy as np


def load_raw_data_to_dict(path):
    """
    loads the data from `path` text file, every line that starts with a 'T' is a list of measurements
    create data dictionary that holds the values as np.array
    :param path: (str) path to original text file from kdd cup 2018 air quality concentration data
    (without missing data, i.e. with 0 instead of missing value)
    :return: data dictionary
    """
    data_dict = dict(Beijing=dict(), London=dict())
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('T'):  # all the relevant lines starts with T, e.g., T1, T2...
            sid, city, station, pollutant, first_measurement_datetime, values = line.strip().split(':')
            values = np.array(values.split(','), dtype=np.float64)
            id_dict = {(sid, pollutant, first_measurement_datetime): values}
            if station not in data_dict[city]:
                data_dict[city][station] = id_dict
            else:
                data_dict[city][station].update(id_dict)

    return data_dict


if __name__ == '__main__':
    path = '[original] kdd_cup_2018_dataset_with_missing_values_as_zeros.txt'
    data_dict = load_raw_data_to_dict(path=path)
