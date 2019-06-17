import numpy as np
import os
import pandas as pd


def func_read_csv(dir_data, extension='.csv'):
    """
    function to load and concatenate all csv files inside directory: dir_data
    :param dir_data: data input directory
    :param extension: file extention being considered
    :return: dfs=data frame saving data
    """

    # check folder existence
    if not os.path.isdir(dir_data):
        sys.exit("Error, folder does not exist.")

    # list all csv files within the folder
    csv_list = []
    for root, dirs, files in os.walk(dir_data, topdown=True):
        for name in files:
            if os.path.splitext(name)[-1] == extension:
                csv_list.append(os.path.join(root, name))

    # read and concatenate all data
    df_list = []
    file_id = 0
    for file in csv_list:
        file_id += 1
        print("loading {} of {} files: {}.".format(file_id,len(csv_list), file))

        # check I/O
        try:
            df = pd.read_csv(file,low_memory=False)
        except Exception as e:
            sys.exit('Error reading {}'.format(filename))

        # concatenate
        df_list.append(df)

    dfs = pd.concat(df_list, sort=False)
    dfs.drop_duplicates(subset=None, keep='first', inplace=True)
    dfs.reset_index(inplace=True,drop=True)
    return dfs


def func_mag_rms(ax, ay, az, g=9.81):
    """
    function to convert xyz-component into magnitude and rms
    :param ax: x-component
    :param ay: y-component
    :param az: z-component
    :param g: constant (e.g. gravitational acceleration if ax, ay, az represent accelerations)
    :return: a_mag = magnitude, a_rms = rms value of linear accelerometer
    """
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    a_rms = a_mag - g
    return a_mag, a_rms


def func_orientation_angles(ax, ay, az):
    """
    function to convert xyz-component into orientation angles
    :param ax:
    :param ay:
    :param az:
    :return: phi, theta
    """
    phi = np.arctan(ay / (np.sqrt(ax**2 + az**2) + 1e-12))
    theta = np.arctan(-ax / (az + 1e-12))
    return phi, theta
