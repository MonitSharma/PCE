# ==============================================================================
# Copyright 2023-* Marco Sciorilli. All Rights Reserved.
# Copyright 2023-* QRC @ Technology Innovation Institute of Abu Dhabi. All Rights Reserved.
# ==============================================================================

import numpy as np

def gradient_free_optimizer(optimizer_name, search_space, n_iter, loss_function):
    import gradient_free_optimizers
    if optimizer_name == 'HillClimbing':
        opt = gradient_free_optimizers.HillClimbingOptimizer(search_space)
    if optimizer_name == 'StochasticHillClimbing':
        opt = gradient_free_optimizers.StochasticHillClimbingOptimizer(search_space)
    if optimizer_name == 'RepulsingHillClimbing':
        opt = gradient_free_optimizers.RepulsingHillClimbingOptimizer(search_space)
    if optimizer_name == 'SimulatedAnnealingClimbing':
        opt = gradient_free_optimizers.SimulatedAnnealingOptimizer(search_space)
    if optimizer_name == 'DownhillSimplexOptimization':
        opt = gradient_free_optimizers.DownhillSimplexOptimizer(search_space)
    if optimizer_name == 'RandomSearch':
        opt = gradient_free_optimizers.RandomSearchOptimizer(search_space)
    if optimizer_name == 'GridSearch':
        opt = gradient_free_optimizers.GridSearchOptimizer(search_space)
    if optimizer_name == 'RandomRestartHillClimbing':
        opt = gradient_free_optimizers.RandomRestartHillClimbingOptimizer(search_space)
    if optimizer_name == 'RandomAnnealing':
        opt = gradient_free_optimizers.RandomAnnealingOptimizer(search_space)
    if optimizer_name == 'PatternSearch':
        opt = gradient_free_optimizers.PatternSearch(search_space)
    if optimizer_name == 'PowellsMethod':
        opt = gradient_free_optimizers.PowellsMethod(search_space)
    if optimizer_name == 'ParallelTempering':
        opt = gradient_free_optimizers.ParallelTemperingOptimizer(search_space)
    if optimizer_name == 'ParticleSwarmOptimization':
        opt = gradient_free_optimizers.ParticleSwarmOptimizer(search_space)
    if optimizer_name == 'SpiralOptimization':
        opt = gradient_free_optimizers.SpiralOptimization(search_space)
    if optimizer_name == 'EvolutionStrategy':
        opt = gradient_free_optimizers.EvolutionStrategyOptimizer(search_space)
    if optimizer_name == 'BayesianOptimization':
        opt = gradient_free_optimizers.BayesianOptimizer(search_space)
    if optimizer_name == 'LipschitzOptimization':
        opt = gradient_free_optimizers.LipschitzOptimizer(search_space)
    if optimizer_name == 'DIRECTalgorithm':
        opt = gradient_free_optimizers.DirectAlgorithm(search_space)
    if optimizer_name == 'TreeofParzenEstimators':
        opt = gradient_free_optimizers.TreeStructuredParzenEstimators(search_space)
    if optimizer_name == 'ForestOptimizer':
        opt = gradient_free_optimizers.ForestOptimizer(search_space)
    opt.search(loss_function, n_iter=n_iter)  # , verbosity=False)
    return opt


def tensorflow_optimizer(loss, initial_parameters, args=(),
                         options=None):
    from qibo.config import log
    import tensorflow as tf
    sgd_options = {
        "nepochs": 100000,
        "nmessage": 1000,
        "optimizer": "Adam",
        "learning_rate": 0.001
    }
    if options is not None:
        sgd_options.update(options)
    vparams = tf.Variable(initial_parameters)
    optimizer = getattr(tf.optimizers, sgd_options["optimizer"])(
        learning_rate=sgd_options["learning_rate"]
    )

    def opt_step():
        with tf.GradientTape() as tape:
            l = loss(vparams, *args)
        grads = tape.gradient(l, [vparams])
        optimizer.apply_gradients(zip(grads, [vparams]))
        return l

    best_param = 0
    wait = 50
    precision = 0.01
    l_best = 10000000
    stop = 0
    nepochs = 0
    total_precision = 0
    for e in range(sgd_options["nepochs"]):
        nepochs += 1
        l = opt_step()
        if l_best - l.numpy() > 0:
            best_param = vparams.numpy()
            total_precision += l_best - l.numpy()
            l_best = l.numpy()
            if total_precision > precision:
                stop = 0
                total_precision = 0
            else:
                stop += 1
        else:
            stop += 1

        if e % sgd_options["nmessage"] == 1:
            log.info("ite %d : loss %f", e, l.numpy())

        if stop == wait:
            break

    return l_best, best_param, nepochs


def adam(objective, derivative, starting_point, n_iter, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    import math
    x = starting_point
    m = [0.0 for _ in range(len(starting_point))]
    v = [0.0 for _ in range(len(starting_point))]

    best_param = 0
    wait = 10
    precision = 0.1
    l_best = 10000000
    nepochs = 0
    stop = 0
    total_precision = 0
    for t in range(n_iter):
        nepochs += 1
        l = objective(x)
        g = derivative(x)
        for i in range(len(starting_point)):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
            mhat = m[i] / (1.0 - beta1 ** (t + 1))
            vhat = v[i] / (1.0 - beta2 ** (t + 1))
            x[i] = x[i] - alpha * mhat / (math.sqrt(vhat) + eps)
        if l_best - l > 0:
            best_param = x
            total_precision += l_best - l
            l_best = l
            if total_precision > precision:
                stop = 0
                total_precision = 0
            else:
                stop += 1
        else:
            stop += 1
        if stop == wait:
            break
    return l_best, best_param, nepochs


def sgd(loss, gradient, start, learn_rate=0.05, beta1=0.9,beta2=0.999, batch_size=None,
        n_iter=10000, tolerance=0.00001, dtype="float64", maxwait=8, epsilon = 1e-08, cut_getter=None):
    # Checking if the gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")

    # Setting up the data type for NumPy arrays
    dtype_ = np.dtype(dtype)

    # Initializing the values of the variables
    vector = np.array(start, dtype=dtype_)

    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")


    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    # Setting the difference to zero for the first iteration
    # Performing the gradient descent loop
    cut_best = 0
    wait = 0
    best_param = np.array(start, dtype=dtype_)
    counter = 0
    means = np.zeros(len(best_param))

    for i in range(n_iter):
        np.random.seed(i)
        first_moment = tolerance + 1e-06
        second_moment = tolerance + 1e-06
        grad = np.array(gradient(vector), dtype_)
        timestep = 0
        while np.linalg.norm(grad, ord=2) >= tolerance:
            timestep +=1
            counter += 1
            # Recalculating the difference

            if batch_size is not None:
                # Generate random indices
                zero_array = np.zeros_like(grad)
                indices_to_keep = np.random.choice(grad.size, batch_size, replace=False)
                # Set the chosen elements to their original values
                zero_array[indices_to_keep] = grad[indices_to_keep]
                # Replace the original array with the modified one
                grad = zero_array

            first_moment=first_moment*beta1 + (1-beta1)*grad
            second_moment = second_moment*beta2 + (1-beta2)*grad**2
            correct_bias_first_moment =first_moment/(1-beta1**timestep)
            correct_bias_second_moment =second_moment/(1-beta2**timestep)
            adjust = learn_rate*correct_bias_first_moment/(np.sqrt(correct_bias_second_moment) +epsilon)
            vector -= adjust
            grad = np.array(gradient(vector), dtype_)
            # if np.linalg.norm(grad, ord=2) > 1 and timestep>200 :
            #     grad = grad/np.linalg.norm(grad, ord=2)
        cut, _ = cut_getter(vector)
        if i == 0:
            perturbation = np.abs(vector - best_param)
        if cut - cut_best > 0:
            best_param = vector
            cut_best = cut
            wait = 0
        else:
            wait += 1
        if wait == maxwait:
            break
        np.random.seed(int(i + wait))
        perturbations = np.random.normal(means, perturbation)
        vector = best_param + perturbations
    l_best = loss(best_param)
    return l_best, best_param, counter


def newtonian(
        loss,
        initial_parameters,
        args=(),
        method="Powell",
        jac=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
):
    from skquant.interop.scipy import imfil, snobfit
    from scipy.optimize import minimize
    if method == 'imfil' or method == 'snobfit':
        method = snobfit
        options['budget'] = options['maxiter']
        options.pop('maxiter')
        m = minimize(
            loss,
            initial_parameters,
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            options=options,
        )
    else:
        m = minimize(
            loss,
            initial_parameters,
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
        )
    return m.fun, m.x, m


