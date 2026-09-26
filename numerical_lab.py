"""Original CS231N teaching experiments, not Stanford assignment solutions.

Requires NumPy. Run ``python numerical_lab.py`` for deterministic checks.
Functions perform no network, filesystem, training, or device effects. Arrays
are float64 teaching objects, not performance-oriented neural-network layers.
"""

import json
import math

import numpy as np


def finite_array(value, name, ndim=None):
    """Convert finite, nonempty numerical data; reject a mismatched rank."""
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(name + ' must be nonempty and finite')
    if ndim is not None and array.ndim != ndim:
        raise ValueError(name + ' has the wrong number of dimensions')
    return array


def softmax_loss_grad(logits, labels):
    """Mean categorical NLL and its logit gradient; rows are observations.

    logits: finite (N, C) array. labels: integer (N,) array in [0, C).
    Returns (scalar loss, (N,C) gradient). No mutation of caller arrays.
    Log-sum-exp avoids exponentiating an absolute large logit.
    """
    z = finite_array(logits, 'logits', 2)
    y = np.asarray(labels)
    if y.shape != (z.shape[0],) or y.dtype.kind not in 'iu':
        raise ValueError('labels must be one integer per observation')
    if np.any(y < 0) or np.any(y >= z.shape[1]):
        raise ValueError('label out of range')
    shifted = z - z.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1))
    loss = np.mean(log_partition - shifted[np.arange(len(y)), y])
    gradient = np.exp(shifted - log_partition[:, None])
    gradient[np.arange(len(y)), y] -= 1
    gradient /= len(y)
    return float(loss), gradient


def central_gradient(function, point, step=1e-6):
    """Central differences of a pure scalar function at a finite array."""
    x = finite_array(point, 'point').copy()
    if not math.isfinite(step) or step <= 0:
        raise ValueError('step must be finite and positive')
    gradient = np.empty_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def two_layer_loss_grad(inputs, first_weights, second_weights, labels):
    """Bias-free tanh network with categorical loss; return both weight VJPs.

    Inputs have shape (N,D), first weights (D,H), second weights (H,C).
    The omitted biases and tanh activation are explicit teaching choices.
    """
    x = finite_array(inputs, 'inputs', 2)
    first = finite_array(first_weights, 'first_weights', 2)
    second = finite_array(second_weights, 'second_weights', 2)
    if x.shape[1] != first.shape[0] or first.shape[1] != second.shape[0]:
        raise ValueError('network matrix dimensions do not agree')
    hidden = np.tanh(x @ first)
    loss, upstream = softmax_loss_grad(hidden @ second, labels)
    second_gradient = hidden.T @ upstream
    hidden_gradient = (upstream @ second.T) * (1 - hidden**2)
    first_gradient = x.T @ hidden_gradient
    return loss, first_gradient, second_gradient


def contrastive_loss_grad(similarities, positive_indices, temperature):
    """Categorical contrastive loss for candidate similarities in each row.

    Gradient is with respect to similarities, including the 1/temperature
    chain-rule factor. Caller determines augmentations and candidate sets.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    scores = finite_array(similarities, 'similarities', 2)
    loss, gradient = softmax_loss_grad(scores / temperature, positive_indices)
    return loss, gradient / temperature


def quadratic_descent(curvature, step, initial, iterations):
    """Iterates for f(x)=curvature*x^2/2; returns x_0 through x_iterations."""
    if not all(math.isfinite(v) for v in (curvature, step, initial)):
        raise ValueError('inputs must be finite')
    if curvature <= 0 or step < 0:
        raise ValueError('positive curvature and nonnegative step required')
    if type(iterations) is not int or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    values = [float(initial)]
    for _ in range(iterations):
        values.append((1 - step * curvature) * values[-1])
    return np.array(values)


def circular_correlation(signal, kernel, stride=1):
    """One-dimensional cross-correlation with circular boundary conditions.

    y[i] = sum_j kernel[j]*signal[(i+j) mod N], then retain every stride-th
    element. This is an equivariance experiment, not zero-padded 2D vision.
    """
    x = finite_array(signal, 'signal', 1)
    k = finite_array(kernel, 'kernel', 1)
    if type(stride) is not int or stride < 1:
        raise ValueError('stride must be a positive integer')
    output = sum(weight * np.roll(x, -j) for j, weight in enumerate(k))
    return output[::stride]


def attention(queries, keys, values, allowed=None):
    """Single-head scaled dot-product attention without dropout/positions.

    Q: (M,D), K: (N,D), V: (N,Dv); optional Boolean allowed mask: (M,N).
    Returns weights (M,N), outputs (M,Dv). Reject an entirely masked row.
    """
    q = finite_array(queries, 'queries', 2)
    k = finite_array(keys, 'keys', 2)
    v = finite_array(values, 'values', 2)
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError('incompatible attention dimensions')
    logits = q @ k.T / math.sqrt(q.shape[1])
    if allowed is not None:
        mask = np.asarray(allowed)
        if mask.dtype.kind != 'b' or mask.shape != logits.shape:
            raise ValueError('allowed must be a matching Boolean mask')
        if not mask.any(axis=1).all():
            raise ValueError('every query requires an allowed key')
        logits = np.where(mask, logits, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, weights @ v


def box_iou(first, second):
    """Intersection/union of positive-area xyxy boxes in continuous units."""
    a = finite_array(first, 'first', 1)
    b = finite_array(second, 'second', 1)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError('boxes need four coordinates')
    if np.any(a[2:] <= a[:2]) or np.any(b[2:] <= b[:2]):
        raise ValueError('boxes must have positive width and height')
    extent = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(extent))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]))
    return intersection / (union - intersection)


def weighted_shard_mean(means, counts):
    """Combine per-shard means, preserving the full-data mean for unequal N."""
    gradients = finite_array(means, 'means', 2)
    n = np.asarray(counts)
    if n.shape != (len(gradients),) or n.dtype.kind not in 'iu':
        raise ValueError('one integer count per shard required')
    if np.any(n <= 0):
        raise ValueError('shard counts must be positive')
    return (gradients * n[:, None]).sum(axis=0) / n.sum()


def standard_normal_kl(mean, log_variance):
    """KL of diagonal N(mean, exp(log_variance)) to N(0,I), in nats."""
    mu = finite_array(mean, 'mean')
    logvar = finite_array(log_variance, 'log_variance')
    if mu.shape != logvar.shape:
        raise ValueError('mean and log variance shapes must match')
    if np.any(logvar > 700):
        raise ValueError('variance exceeds the float64 teaching range')
    return float(0.5 * np.sum(mu**2 + np.expm1(logvar) - logvar))


def forward_diffusion(clean, noise, alpha_bar):
    """Return sqrt(alpha_bar)*clean + sqrt(1-alpha_bar)*noise.

    Inputs share shape. Caller supplies noise, making randomness explicit.
    Gaussian marginal claims require independent standard-normal noise.
    """
    x = finite_array(clean, 'clean')
    epsilon = finite_array(noise, 'noise')
    if x.shape != epsilon.shape:
        raise ValueError('clean and noise shapes must match')
    if not math.isfinite(alpha_bar) or not 0 <= alpha_bar <= 1:
        raise ValueError('alpha_bar must be in [0, 1]')
    return math.sqrt(alpha_bar) * x + math.sqrt(1 - alpha_bar) * epsilon


def pinhole(points, focal_length):
    """Project positive-depth (N,3) points in focal-length units."""
    xyz = finite_array(points, 'points', 2)
    if xyz.shape[1] != 3 or np.any(xyz[:, 2] <= 0):
        raise ValueError('points must have three coordinates and positive Z')
    if not math.isfinite(focal_length) or focal_length <= 0:
        raise ValueError('focal length must be finite and positive')
    return focal_length * xyz[:, :2] / xyz[:, 2, None]


def check_all():
    """Numerical checks and counterexamples; raise on any unmet assertion."""
    report = {}
    logits = np.array([[1.2, -0.7, 0.2], [-0.4, 0.8, 1.5]])
    labels = np.array([0, 2])
    original = logits.copy()
    loss, analytical = softmax_loss_grad(logits, labels)
    numerical = central_gradient(lambda z: softmax_loss_grad(z, labels)[0],
                                 logits)
    error = float(np.max(np.abs(analytical - numerical)))
    assert error < 1e-8
    assert np.array_equal(original, logits)
    assert np.allclose(analytical.sum(axis=1), 0, atol=1e-15)
    shifted_loss, shifted_grad = softmax_loss_grad(logits + 10000, labels)
    assert abs(shifted_loss - loss) < 1e-11
    assert np.allclose(shifted_grad, analytical, atol=1e-11)
    assert softmax_loss_grad([[1000., -1000.]], [1])[0] == 2000
    report['softmax_gradient_max_absolute_error'] = error

    x = np.array([[1., 0.4], [-0.2, 0.7]])
    first = np.array([[0.2, -0.3, 0.4], [0.5, 0.1, -0.2]])
    second = np.array([[0.3, -0.5], [0.6, 0.2], [-0.1, 0.4]])
    y = np.array([0, 1])
    _, first_grad, second_grad = two_layer_loss_grad(x, first, second, y)
    first_fd = central_gradient(
        lambda w: two_layer_loss_grad(x, w, second, y)[0], first)
    second_fd = central_gradient(
        lambda w: two_layer_loss_grad(x, first, w, y)[0], second)
    backprop_error = max(float(np.max(abs(first_fd - first_grad))),
                         float(np.max(abs(second_fd - second_grad))))
    assert backprop_error < 1e-8
    report['two_layer_backprop_max_absolute_error'] = backprop_error

    similarities = np.array([[0.7, 0.1, -0.2], [-0.1, 0.8, 0.5]])
    _, contrastive_grad = contrastive_loss_grad(similarities, y, 0.3)
    contrastive_fd = central_gradient(
        lambda s: contrastive_loss_grad(s, y, 0.3)[0], similarities)
    contrastive_error = float(np.max(abs(contrastive_grad - contrastive_fd)))
    assert contrastive_error < 1e-8
    report['contrastive_temperature_chain_rule_error'] = contrastive_error

    clip = np.array([[0., 0.], [1., 0.], [2., 0.]])
    reversed_clip = clip[::-1]
    assert np.array_equal(clip.mean(axis=0), reversed_clip.mean(axis=0))
    assert not np.array_equal(clip[-1] - clip[0],
                              reversed_clip[-1] - reversed_clip[0])
    report['temporal_mean_loses_motion_direction_counterexample'] = 'passed'

    stable = quadratic_descent(4, 0.4, 1, 20)
    unstable = quadratic_descent(4, 0.6, 1, 20)
    assert abs(stable[-1]) < 0.0001 and abs(unstable[-1]) > 800
    assert quadratic_descent(4, 0.5, 1, 4).tolist() == [1, -1, 1, -1, 1]
    report['gradient_descent_final_stable_unstable'] = [
        stable[-1], unstable[-1]]

    signal, kernel = np.arange(8.)**2, np.array([1., -2., 1.])
    correlated = circular_correlation(signal, kernel)
    assert np.allclose(circular_correlation(np.roll(signal, 1), kernel),
                       np.roll(correlated, 1))
    a = circular_correlation(np.roll(signal, 1), kernel, 2)
    b = circular_correlation(signal, kernel, 2)
    assert not any(np.allclose(a, np.roll(b, j)) for j in range(len(b)))
    report['convolution_equivariance_and_stride_counterexample'] = 'passed'

    rng = np.random.default_rng(231)
    q, k, v = rng.normal(size=(3, 4)), rng.normal(
        size=(5, 4)), rng.normal(size=(5, 2))
    weights, output = attention(q, k, v)
    permutation = [3, 1, 4, 0, 2]
    _, permuted = attention(q, k[permutation], v[permutation])
    assert np.allclose(output, permuted)
    assert np.allclose(weights.sum(axis=1), 1)
    causal, _ = attention(q, q, q, np.tril(np.ones((3, 3), dtype=bool)))
    assert np.all(causal[np.triu_indices(3, 1)] == 0)
    report['attention_key_value_permutation_max_error'] = float(
        np.max(abs(output - permuted)))

    assert abs(box_iou([0, 0, 2, 2], [1, 1, 3, 3]) - 1 / 7) < 1e-15
    assert box_iou([0, 0, 1, 1], [1, 0, 2, 1]) == 0
    assert box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    report['box_iou_partial_disjoint_identity'] = 'passed'

    samples = np.array([[1., 4.], [3., 2.], [8., -1.], [12., 7.]])
    shard_means = np.vstack((samples[:1].mean(0), samples[1:].mean(0)))
    combined = weighted_shard_mean(shard_means, [1, 3])
    assert np.allclose(combined, samples.mean(0))
    assert not np.allclose(shard_means.mean(0), samples.mean(0))
    report['unequal_shard_mean_unweighted_error'] = float(
        np.max(abs(shard_means.mean(0) - samples.mean(0))))

    assert standard_normal_kl([0, 0], [0, 0]) == 0
    assert standard_normal_kl([1, 2], [0, 0]) == 2.5
    assert standard_normal_kl([0], [math.log(4)]) > 0
    report['diagonal_gaussian_kl_boundary_cases'] = 'passed'

    clean, noise = np.full(200000, 2.), rng.normal(size=200000)
    noised = forward_diffusion(clean, noise, 0.36)
    assert abs(noised.mean() - 1.2) < 0.006
    assert abs(noised.var() - 0.64) < 0.008
    assert np.array_equal(forward_diffusion(clean, noise, 1), clean)
    assert np.array_equal(forward_diffusion(clean, noise, 0), noise)
    report['diffusion_empirical_mean_variance'] = [
        float(noised.mean()), float(noised.var())]

    points = np.array([[1., 2., 4.], [-2., 1., 5.]])
    assert np.allclose(pinhole(points, 100), [[25, 50], [-40, 20]])
    assert np.allclose(pinhole(7 * points, 100), pinhole(points, 100))
    report['pinhole_projection_scale_ambiguity'] = 'passed'

    rejected = 0
    invalid = [lambda: softmax_loss_grad([[1, 2]], [2]),
               lambda: softmax_loss_grad([[1, float('nan')]], [0]),
               lambda: attention([[1]], [[1]], [[1]], [[False]]),
               lambda: weighted_shard_mean([[1], [2]], [0, 1]),
               lambda: forward_diffusion([1], [0], 1.1),
               lambda: pinhole([[0, 0, 0]], 1),
               lambda: box_iou([0, 0, 0, 1], [0, 0, 1, 1])]
    for check in invalid:
        try:
            check()
        except ValueError:
            rejected += 1
    assert rejected == len(invalid)
    report['invalid_inputs_rejected'] = rejected
    return report


if __name__ == '__main__':
    print(json.dumps(check_all(), indent=2))
