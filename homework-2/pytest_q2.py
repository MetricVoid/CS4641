import pytest
import numpy as np
from typing import Dict, Tuple
from hw2 import *

def prepend_col_1s(arr):
    return np.insert(arr, 0, 1,axis=1)
@pytest.fixture(scope='module', autouse=True)
def all_data() -> Dict[str, Tuple[np.array, np.array]]:
    return {
        'exp_samp': load_data('1D-exp-samp.txt'),
        'exp_uni': load_data('1D-exp-uni.txt'),
        'no_noise_lin': load_data('1D-no-noise-lin.txt'),
        'quad_uni_noise': load_data('1D-quad-uni-noise.txt'),
        'quad_uni': load_data('1D-quad-uni.txt'),
        'noisy_lin': load_data('2D-noisy-lin.txt'),
    }

@pytest.fixture(scope='module', autouse=True)
def data_1d() -> Dict[str, Tuple[np.array, np.array]]:
    return {
        'exp_samp': load_data('1D-exp-samp.txt'),
        'exp_uni': load_data('1D-exp-uni.txt'),
        'no_noise_lin': load_data('1D-no-noise-lin.txt'),
        'quad_uni_noise': load_data('1D-quad-uni-noise.txt'),
        'quad_uni': load_data('1D-quad-uni.txt'),
    }

@pytest.fixture(scope='module', autouse=True)
def data_2d() -> Dict[str, Tuple[np.array, np.array]]:
    return {
        'noisy_lin': load_data('2D-noisy-lin.txt'),
    }

def test_init(all_data):
    pass

def test_linreg_closed_form(all_data):
    for file_name, (X, Y) in all_data.items():
        try:
            np.testing.assert_allclose(linreg_closed_form(X, Y), np.linalg.lstsq(prepend_col_1s(X), Y, rcond=None)[0], atol=1e-7)
        except AssertionError as e:
            print(linreg_closed_form(X, Y))
            print(file_name)
            raise(e)

def test_linreg_closed_form_singular(all_data):
    # Test whether linreg_closed_form works for singular matrices. Trying to find the regular inverse of a singular matrix fails, so we should find the pseudo-inverse in this case.
    for file_name, (X, Y) in all_data.items():
        try:
            # Duplicate a column in X
            X = np.insert(X, 1, X[:, 0], axis=1)
            np.testing.assert_allclose(linreg_closed_form(X, Y), np.linalg.lstsq(prepend_col_1s(X), Y, rcond=None)[0], atol=1e-7)
        except AssertionError as e:
            print(linreg_closed_form(X, Y))
            print(file_name)
            raise(e)

def test_loss(all_data):
    for file_name, (X, Y) in all_data.items():
        try:
            θ, residuals, _, _ = np.linalg.lstsq(prepend_col_1s(X), Y, rcond=None)
            _loss = loss(θ, X, Y)
            assert type(_loss) in [float, np.float16, np.float32, np.float64]
            np.testing.assert_allclose(_loss, residuals / (2 * X.shape[0]), atol=1e-7)
        except AssertionError as e:
            print(loss(θ, X, Y))
            print(file_name)
            raise(e)

@pytest.mark.parametrize(
    'init_Theta_val, alpha, num_iters, print_iters',
    [(0.5, 0.05, 5000, False)]
)
def test_linreg_grad_desc(all_data, init_Theta_val, alpha, num_iters, print_iters):
    for file_name, (X, Y) in all_data.items():
        try:
            θ = np.linalg.lstsq(prepend_col_1s(X), Y, rcond=None)[0]
            θ_init = np.full((X.shape[1] + 1, 1), init_Theta_val)
            step_history = linreg_grad_desc(θ_init, X, Y, alpha=alpha, num_iters=num_iters, print_iters=print_iters)
            θ_hat, error = step_history[-1]
            np.testing.assert_allclose(θ_hat, θ, atol=1e-5)
            assert error == loss(θ_hat, X, Y)
        except AssertionError as e:
            with open('errors.txt', 'w+') as f:
                f.write(str(step_history))
            print(file_name)
            raise(e)

@pytest.mark.parametrize(
    'num_fourier_features, alpha, num_iters, print_iters',
    [(10, 0.05, 5000, False)]
)
def test_random_fourier_features(all_data, num_fourier_features, alpha, num_iters, print_iters):
    for file_name, (X, Y) in all_data.items():
        try:
            N, D = X.shape
            θ_hat, Ω, B = random_fourier_features(X, Y, num_fourier_features, alpha, num_iters, print_iters)
            assert θ_hat.shape in [(num_fourier_features+1,), (num_fourier_features+1, 1)]
            assert Ω.shape == (D, num_fourier_features)
            assert B.shape == (num_fourier_features,)
            Φ = apply_RFF_transform(X, Ω, B)
            # np.linalg.lstsq produces ridiculously high weights, probably due to overfitting, so instead we'll compare our RFF to gradient descent.
            initial_Theta = (numpy.random.random(size=(num_fourier_features+1,1))-0.5)*0.2
            θ = linreg_grad_desc(initial_Theta, Φ, Y, alpha, num_iters, print_iters)[-1][0]
            # θ = np.ndarray.flatten(np.linalg.lstsq(prepend_col_1s(Φ), Y)[0])
            np.testing.assert_allclose(θ_hat, θ, atol=0.2)
        except AssertionError as e:
            with open('errors.txt', 'w+', encoding='utf-8') as f:
                f.write('RFFθ: {}\n'.format(θ_hat))
                f.write('Ω: {}\n'.format(Ω))
                f.write('B: {}\n'.format(B))
            print(file_name)
            raise(e)