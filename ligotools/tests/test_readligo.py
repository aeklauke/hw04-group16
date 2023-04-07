from ligotools import readligo as rl
import numpy as np
import pytest

fn_H1 = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
fn_L1 = "data/L-L1_LOSC_4_V2-1126259446-32.hdf5"
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')


def test_load_data_h1():
    assert type(strain_H1) == np.ndarray
    assert type(time_H1) == np.ndarray
    assert type(chan_dict_H1) == dict
     
def test_load_data_l1():
    assert type(strain_H1) == np.ndarray
    assert type(time_H1) == np.ndarray
    assert type(chan_dict_H1) == dict

def test_length_h1():
    assert len(strain_H1) == 131072
    assert len(time_H1) == 131072
    assert len(chan_dict_H1) == 13

def test_length_l1():
    assert len(strain_L1) == 131072
    assert len(time_L1) == 131072
    assert len(chan_dict_L1) == 13
