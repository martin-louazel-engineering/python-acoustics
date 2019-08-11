import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest

from acoustics.room import (mean_alpha, nrc, t60_sabine, t60_eyring, t60_millington, t60_fitzroy, t60_arau, t60_impulse,
                            c50_from_file, c80_from_file, d50_from_file, d80_from_file, centre_time, st_early_from_file,
                            st_late_from_file, st_total_from_file)
from acoustics.bands import octave, third

import sys
sys.path.append('..')
from get_data_path import data_path


def setup_module(room):
    room.surfaces = np.array([240, 600, 500])
    room.alpha = np.array([0.1, 0.25, 0.45])
    room.alpha_bands = np.array([[0.1, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.45, 0.45, 0.45, 0.45]])
    room.volume = 3000


def test_t60_sabine():
    calculated = t60_sabine(surfaces, alpha, volume)
    real = 1.211382149
    assert_almost_equal(calculated, real)


def test_t60_sabine_bands():
    calculated = t60_sabine(surfaces, alpha_bands, volume)
    real = np.array([1.211382149, 1.211382149, 1.211382149, 1.211382149])
    assert_array_almost_equal(calculated, real)


def test_t60_eyring():
    calculated = t60_eyring(surfaces, alpha, volume)
    real = 1.020427763
    assert_almost_equal(calculated, real)


def test_t60_eyring_bands():
    calculated = t60_eyring(surfaces, alpha_bands, volume)
    real = np.array([1.020427763, 1.020427763, 1.020427763, 1.020427763])
    assert_array_almost_equal(calculated, real)


def test_t60_millington():
    calculated = t60_millington(surfaces, alpha, volume)
    real = 1.020427763
    assert_almost_equal(calculated, real)


def test_t60_millington_bands():
    calculated = t60_millington(surfaces, alpha_bands, volume)
    real = np.array([1.020427763, 1.020427763, 1.020427763, 1.020427763])
    assert_array_almost_equal(calculated, real)


def test_t60_fitzroy():
    surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
    alpha_fitzroy = np.array([0.1, 0.1, 0.25, 0.25, 0.45, 0.45])
    calculated = t60_fitzroy(surfaces_fitzroy, alpha_fitzroy, volume)
    real = 0.699854185
    assert_almost_equal(calculated, real)


def test_t60_fitzroy_bands():
    surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
    alpha_bands_f = np.array([[0.1, 0.1, 0.25, 0.25, 0.45, 0.45], [0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
                              [0.1, 0.1, 0.25, 0.25, 0.45, 0.45]])
    calculated = t60_fitzroy(surfaces_fitzroy, alpha_bands_f, volume)
    real = np.array([0.699854185, 0.699854185, 0.699854185])
    assert_array_almost_equal(calculated, real)


def test_t60_arau():
    Sx = surfaces[0]
    Sy = surfaces[1]
    Sz = surfaces[2]
    calculated = t60_arau(Sx, Sy, Sz, alpha, volume)
    real = 1.142442931
    assert_almost_equal(calculated, real)


def test_t60_arau_bands():
    Sx = surfaces[0]
    Sy = surfaces[1]
    Sz = surfaces[2]
    calculated = t60_arau(Sx, Sy, Sz, alpha_bands, volume)
    real = np.array([1.142442931, 1.142442931, 1.142442931, 1.142442931])
    assert_array_almost_equal(calculated, real)


def test_mean_alpha_float():
    alpha = 0.1
    surface = 10
    calculated = mean_alpha(alpha, surface)
    real = 0.1
    assert_almost_equal(calculated, real)


def test_mean_alpha_1d():
    alpha = np.array([0.1, 0.2, 0.3])
    surfaces = np.array([20, 30, 40])
    calculated = mean_alpha(alpha, surfaces)
    real = 0.222222222
    assert_almost_equal(calculated, real)


def test_mean_alpha_bands():
    alpha = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    surfaces = np.array([20, 30, 40])
    calculated = mean_alpha(alpha, surfaces)
    real = np.array([0.222222222, 0.222222222, 0.222222222])
    assert_array_almost_equal(calculated, real)


def test_nrc_1d():
    alpha = np.array([0.1, 0.25, 0.5, 0.9])
    calculated = nrc(alpha)
    real = 0.4375
    assert_almost_equal(calculated, real)


def test_nrc_2d():
    alphas = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]])
    calculated = nrc(alphas)
    real = np.array([0.25, 0.55])
    assert_array_almost_equal(calculated, real)


@pytest.mark.parametrize("file_name, bands, rt, expected", [
    (data_path() + 'ir_sportscentre_omni.wav', octave(125, 4000), 't30',
     np.array([7.388, 8.472, 6.795, 6.518, 4.797, 4.089])),
    (data_path() + 'ir_sportscentre_omni.wav', octave(125, 4000), 'edt',
     np.array([4.667, 5.942, 6.007, 5.941, 5.038, 3.735])),
    (data_path() + 'living_room_1.wav', octave(63, 8000), 't30',
     np.array([0.274, 0.365, 0.303, 0.259, 0.227, 0.211, 0.204, 0.181])),
    (data_path() + 'living_room_1.wav', octave(63, 8000), 't20',
     np.array([0.300, 0.365, 0.151, 0.156, 0.102, 0.076, 0.146, 0.152])),
    (data_path() + 'living_room_1.wav', octave(63, 8000), 't10',
     np.array([0.185, 0.061, 0.109, 0.024, 0.039, 0.023, 0.105, 0.071])),
    (data_path() + 'living_room_1.wav', octave(63, 8000), 'edt',
     np.array([0.267, 0.159, 0.080, 0.037, 0.021, 0.010, 0.022, 0.020])),
    (data_path() + 'living_room_1.wav', third(100, 5000), 't30',
     np.array([
         0.318, 0.340, 0.259, 0.311, 0.267, 0.376, 0.342, 0.268, 0.212, 0.246, 0.211, 0.232, 0.192, 0.231, 0.252, 0.202,
         0.184, 0.216
     ])),
    (data_path() + 'living_room_1.wav', third(100, 5000), 't20',
     np.array([
         0.202, 0.383, 0.189, 0.173, 0.141, 0.208, 0.323, 0.221, 0.102, 0.110, 0.081, 0.128, 0.072, 0.074, 0.087, 0.129,
         0.137, 0.171
     ])),
    (data_path() + 'living_room_1.wav', third(100, 5000), 't10',
     np.array([
         0.110, 0.104, 0.132, 0.166, 0.135, 0.040, 0.119, 0.223, 0.025, 0.023, 0.047, 0.050, 0.010, 0.017, 0.039, 0.084,
         0.154, 0.093
     ])),
    (data_path() + 'living_room_1.wav', third(100, 5000), 'edt',
     np.array([
         0.354, 0.328, 0.284, 0.210, 0.132, 0.116, 0.085, 0.114, 0.064, 0.045, 0.047, 0.047, 0.024, 0.017, 0.016, 0.022,
         0.020, 0.036
     ])),
])
def test_t60_impulse(file_name, bands, rt, expected):
    calculated = t60_impulse(file_name, bands, rt)
    assert_array_almost_equal(calculated, expected, decimal=0)


@pytest.mark.parametrize("file_name, bands, expected", [
    (data_path() + 'living_room_1.wav', octave(63, 8000), np.array([8., 18., 23., 26., 30., 31., 27., 29.])),
    (data_path() + 'living_room_1.wav', third(100, 5000),
     np.array([3., 6., 7., 13., 18., 23., 20., 19., 28., 30., 30., 27., 32., 31., 30., 28., 29., 25.])),
])
def test_c50_from_file(file_name, bands, expected):
    calculated = c50_from_file(file_name, bands)
    assert_array_almost_equal(calculated, expected, decimal=0)


@pytest.mark.parametrize("file_name, bands, expected", [
    (data_path() + 'living_room_1.wav', octave(63, 8000),
     np.array([18.542, 23.077, 27.015, 31.743, 35.469, 36.836, 33.463, 36.062])),
    (data_path() + 'living_room_1.wav', third(100, 5000),
     np.array([17., 14., 17., 24., 26., 27., 22., 26., 34., 35., 34., 32., 38., 38., 34., 34., 35., 32.])),
])
def test_c80_from_file(file_name, bands, expected):
    calculated = c80_from_file(file_name, bands)
    assert_array_almost_equal(calculated, expected, decimal=0)


@pytest.mark.parametrize("file_name, bands, expected", [
    (data_path() + 'living_room_1.wav', octave(63, 8000),
     np.array([0.85, 0.98, 1., 1., 1., 1., 1., 1.]),
    (data_path() + 'living_room_1.wav', third(100, 5000),
     np.array([0.64, 0.8 , 0.83, 0.95, 0.98, 0.99, 0.99, 0.99, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])),
])
def test_d50_from_file(file_name, bands, expected):
    calculated = d50_from_file(file_name, bands)
    assert_array_almost_equal(calculated, expected, decimal=2)


@pytest.mark.parametrize("file_name, bands, expected", [
    (data_path() + 'living_room_1.wav', octave(63, 8000),
     np.array([0.98, 0.99, 1., 1., 1., 1., 1., 1.]),
    (data_path() + 'living_room_1.wav', third(100, 5000),
     np.array([0.98, 0.96, 0.98, 1., 1., 1., 0.99, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])),
])
def test_d80_from_file(file_name, bands, expected):
    calculated = d80_from_file(file_name, bands)
    assert_array_almost_equal(calculated, expected, decimal=2)


@pytest.mark.parametrize("file_name, bands, expected", [
    (data_path() + 'living_room_1.wav', octave(63, 8000),
     np.array([17.79, 3.8 , -10.53, -18.21, -22.58, -25.79, -17.85, -19.32]),
    (data_path() + 'living_room_1.wav', third(100, 5000),
     np.array([39.91, 33.59, 21.14, 20.95, 9.74, 1.39, -5.21, -6.08, -16.4, -22.86, -20.7, -17.53,
               -25.3, -26.28, -24.3, -18.24, -18., -16.87])),
])
def test_st_early_from_file(file_name, bands, expected):
    calculated = st_early_from_file(file_name, bands)
    assert_array_almost_equal(calculated, expected, decimal=2)


@pytest.mark.parametrize("file_name, bands, expected", [
    (data_path() + 'living_room_1.wav', octave(63, 8000),
     np.array([-0.56, -16.95, -28.54, -35.46, -39.5 , -39.99, -36.82, -39.63]),
    (data_path() + 'living_room_1.wav', third(100, 5000),
     np.array([19.69, 15.1 , -0.83, -3.37, -15.75, -19.6, -22.88, -26.83, -36.9, -38.36, -39.08,
               -36.82, -40.55, -42.07, -37.02, -37.57, -38.8, -35.19])),
])
def test_st_late_from_file(file_name, bands, expected):
    calculated = st_late_from_file(file_name, bands)
    assert_array_almost_equal(calculated, expected, decimal=3)


@pytest.mark.parametrize("file_name, bands, expected", [
    (data_path() + 'living_room_1.wav', octave(63, 8000),
     np.array([17.85, 3.84, -10.46, -18.13, -22.49, -25.62, -17.79, -19.28]),
    (data_path() + 'living_room_1.wav', third(100, 5000),
     np.array([39.95, 33.65, 21.17, 20.96, 9.75, 1.42, -5.13, -6.04, -16.36, -22.74, -20.63,
               -17.48, -25.17, -26.17, -24.07, -18.19, -17.96, -16.8])),
])
def test_st_total_from_file(file_name, bands, expected):
    calculated = st_total_from_file(file_name, bands)
    assert_array_almost_equal(calculated, expected, decimal=3)


def teardown_module(room):
    pass
