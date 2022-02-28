import os
from functools import reduce

import numpy as np
import scipy
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras


def summarize_diagnostics(history):
    """
    # Plot diagnostic learning curves
    :param history: Keras history object
    :return: Show plots
    """
    # Plot loss
    plt.figure(figsize=(10, 10), dpi=80)
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history['loss'], color='blue', label='train')
    plt.plot(history['val_loss'], color='orange', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.legend()
    # Plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history['accuracy'], color='blue', label='train')
    plt.plot(history['val_accuracy'], color='orange', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracies_by_param(model_state_by_type, param_name, filename, ylim_left=None, ylim_right=None):
    """
    Given a set of parameter values (e.g. batch sizes) and histories, this function
    creates two plots: one of training accuracy and another of validation accuracy
    :param param_values: List of parameter values used to generate histories (e.g. batch sizes)
    :param history_dict: Dictionary from param value to a Keras history.history
    :param param_name: String name of the parameter (e.g. 'batch size')
    :param filename: file to save the plot to
    """
    plt.figure(figsize=(20, 7), dpi=80)
    plt.subplot(121)
    plt.title('Effect of {} on training accuracy'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['accuracy'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(ylim_left, ylim_right)
        plt.grid(True)
        plt.legend(loc='best')

    plt.subplot(122)
    plt.title('Effect of {} on validation accuracy'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['val_accuracy'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(ylim_left, ylim_right)
        plt.grid(True)
        plt.legend(loc='best')

    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + '{}.png'.format(filename), format="png")
    plt.show()


def plot_loss_by_param(model_state_by_type, param_name, filename, ylim_left=None, ylim_right=None):
    """
    Given a set of parameter values (e.g. batch sizes) and histories, this function
    creates two plots: one of training loss and another of validation loss
    :param param_values: List of parameter values used to generate histories (e.g. batch sizes)
    :param history_dict: Dictionary from param value to a Keras history.history
    :param param_name: String name of the parameter (e.g. 'batch size')
    :param filename: file to save the plot to
    """
    plt.figure(figsize=(20, 7), dpi=80)
    plt.subplot(121)
    plt.title('Effect of {} on training loss'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['loss'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(ylim_left, ylim_right)
        plt.grid(True)
        plt.legend(loc='best')

    plt.subplot(122)
    plt.title('Effect of {} on validation loss'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['val_loss'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(ylim_left, ylim_right)
        plt.grid(True)
        plt.legend(loc='best')

    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + '{}.png'.format(filename), format="png")
    plt.show()


def plot_generalization_gap_by_param(model_state_by_type, param_name, filename, clipping_val=None, ylim_left=None,
                                     ylim_right=None):
    """
    Given a set of parameter values (e.g. batch sizes) and histories, this function
    creates one plot representing generalization gap: val_loss/train_loss
    :param param_values: List of parameter values used to generate histories (e.g. batch sizes)
    :param history_dict: Dictionary from param value to a Keras history.history
    :param param_name: String name of the parameter (e.g. 'batch size')
    :param filename: file to save the plot to
    """
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Effect of {} on generalization gap'.format(param_name))
    for typ, state in model_state_by_type.items():
        gen_gap = np.array(state.history['val_loss']) / np.array(state.history['loss'])
        if clipping_val:
            gen_gap = np.clip(gen_gap, None, clipping_val)
        plt.plot(gen_gap, label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Genralization Gap')
        plt.ylim(ylim_left, ylim_right)
        plt.grid(True)
        plt.legend(loc='best')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + '{}.png'.format(filename), format="png")
    plt.show()


def visualize_weights(weights_by_key, filename, bins=None):
    if not bins:
        bins = [0.005 * a - 0.3 for a in range(120)]
    plt.figure(figsize=(10, 10), dpi=80)
    plt.title('Distribution of weights by model')

    for model, weight in weights_by_key.items():
        flat_weight = np.ndarray.flatten(weight)
        max_wt = np.max(flat_weight)
        min_wt = np.max(flat_weight)

        print('Model: {model}, Max Weight: {max_wt}, Min Weight: {min_wt}'.format(**locals()))
        sns.distplot(flat_weight, label=str(model), kde=False, bins=bins, )
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend(loc='best')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + '{}.png'.format(filename), format="png")
    plt.show()


def get_num_elems_from_shape(shape):
    """
    Given the shape of a vector, returns how many elements are in the vector
    :param shape: shape of vector
    :return: number of elements
    """
    return reduce(lambda x, y: x * y, shape)


def unflatten_weights(flattened_weights, weight_shapes):
    """
    Given a flattened vector of weights, and the desired shapes for each layer, this fn reshapes the weights.
    :param flattened_weights: 1-dimensional vector containing weight values
    :param weight_shapes: list of shapes (one per layer in model)
    :return: list containing reshaped weight vectors (one per layer in model)
    """
    if len(flattened_weights) != sum([get_num_elems_from_shape(shape) for shape in weight_shapes]):
        print("Weight shapes do not match number of flattened weights!")
    i = 0
    unflattened_weights = []
    for shape in weight_shapes:
        num_elems = get_num_elems_from_shape(shape)
        reshaped_weights = np.reshape(flattened_weights[i:i + num_elems], shape)
        unflattened_weights.append(reshaped_weights)
        i += num_elems
    return unflattened_weights


def get_negative_loss(flattened_weights, *args):
    """
    This function sets the last layer of the model to the weights provided, then computes the negative loss.
    This is used as a helper function for get_sharpness.

    :param flattened_weights: 1-dimensional vector containing weight values
    :param *args: (model, data, weight_shapes)
    :return: negative loss of model evaluated on the data provided
    """
    model, data, weight_shapes = args
    unflattened_weights = unflatten_weights(flattened_weights, weight_shapes)
    model.set_weights(unflattened_weights)
    loss, accuracy = model.evaluate(data)
    return -loss


def get_negative_loss_gradient(flattened_weights, *args):
    """
    Computes the gradient of the negative loss with respect to the model weights.

    :param flattened_weights: flattened model weights
    :param *args: (model, data, weight_shapes)
    :return: flattened gradient with respect to weights (1d vector)
    """
    model, data, weight_shapes = args
    unflattened_weights = unflatten_weights(flattened_weights, weight_shapes)
    model.set_weights(unflattened_weights)

    batch_gradients = []
    for x, y in data:
        with tf.GradientTape() as tape:
            preds = model(x)
            negative_loss = tf.math.negative(
                tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, preds)))

        gradients = [tf.cast(g, tf.float64).numpy() for g in tape.gradient(negative_loss, model.trainable_variables)]
        flattened_gradients = np.concatenate([g.flatten() for g in gradients])
        batch_gradients.append(flattened_gradients)

    return np.sum(batch_gradients, axis=0)


def get_sharpness(model, data, epsilon=1e-2):
    """
    This function computes the sharpness of a minimizer by maximizing the loss in a neighborhood around the minimizer.
    Based on sharpness metric defined in https://arxiv.org/pdf/1609.04836.pdf.

    :param model: model, where the weights represent a minimizer of the loss function
    :param data: data to evaluate the model on
    :param epsilon: controls the size of the neighborhood to explore
    :return: sharpness
    """
    # Get original loss
    original_loss, original_accuracy = model.evaluate(data)

    # Compute bounds on weights
    weights = model.get_weights()
    weight_shapes = [w.shape for w in weights]
    flattened_weights = np.concatenate([x.flatten() for x in weights])
    delta = epsilon * (np.abs(flattened_weights) + 1)
    lower_bounds = flattened_weights - delta
    upper_bounds = flattened_weights + delta

    # Create copy of model so we don't modify original
    path = 'tmp/sharpness_model_clone.h5'
    # os.makedirs(path, exist_ok=True)
    model.save(path)
    model_clone = keras.models.load_model(path)
    os.remove(path)

    # Minimize
    x, f, d = scipy.optimize.fmin_l_bfgs_b(
        func=get_negative_loss,
        fprime=get_negative_loss_gradient,
        x0=flattened_weights,
        args=(model_clone, data, weight_shapes),
        bounds=list(zip(lower_bounds, upper_bounds)),
        maxiter=10,
        maxls=1,
        disp=1,
    )

    # Compute sharpness
    sharpness = (-f - original_loss) / (1 + original_loss) * 100
    return sharpness


# Based on https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py#L195
def get_random_filter_normalized_direction(weights):
    """
    Given a set of weights for a model, returns a random Gaussian direction.
    Normalize each convolutional filter or each FC neuron to match the corresponding norm in the weights parameter.
    :param weights: model weights
    """
    random_direction = []
    for w in weights:
        num_dimensions = len(w.shape)

        # For biases, set to 0
        if num_dimensions == 1:
            new_w = np.zeros(w.shape)

        # For fully-connected layers, generate random vector for each neuron and normalize
        elif num_dimensions == 2:
            new_w = np.random.randn(*w.shape)
            for f in range(w.shape[-1]):
                new_filter = new_w[:, f]
                old_filter = w[:, f]
                new_filter *= np.linalg.norm(old_filter) / np.linalg.norm(new_filter)

        # For convolutional layers, generate random vector for each filter and normalize
        elif num_dimensions == 4:
            new_w = np.random.randn(*w.shape)
            for f in range(w.shape[-1]):
                new_filter = new_w[:, :, :, f]
                old_filter = w[:, :, :, f]
                new_filter *= np.linalg.norm(old_filter) / np.linalg.norm(new_filter)

        random_direction.append(new_w)
    return random_direction


def plot_loss_visualization_1d(base_model, training_data, validation_data, build_model_function, title=None,
                               output_filename=None):
    """
    Visualizes the minimizer for a model along a random Gaussian filter-normalized direction.
    :param build_model_function: function which builds a new non-trained model
    :param base_model: model to evaluate
    :param training_data: training data, used to generate training loss numbers
    :param validation_data: validation data, used to generates validation loss numbers
    :param title: title for the plot
    :param output_filename: file to save the plot to
    :return: x_values, train_losses, validation_losses
    """
    # Get weights and generate random direction
    weights = base_model.get_weights()
    direction = get_random_filter_normalized_direction(weights)

    # Set up new model and plotting variables
    x_values = np.linspace(-1, 1, 20)
    train_losses = []
    validation_losses = []
    new_model = build_model_function()

    # Compute training and validation loss for each linear combination of weight and direction
    for x in x_values:
        print("\nx: ", x)

        # Compute and set weights
        new_weights = [w + x * d for w, d in zip(weights, direction)]
        new_model.set_weights(new_weights)

        # Evaluate model
        train_loss, train_accuracy = new_model.evaluate(training_data)
        validation_loss, validation_accuracy = new_model.evaluate(validation_data)

        # Store losses
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

    # Plot results
    plt.plot(x_values, train_losses, linestyle='solid', label='train')
    plt.plot(x_values, validation_losses, linestyle='dashed', label='validation')
    plt.ylabel('Loss')
    plt.xlabel('Alpha')
    plt.legend()
    if title:
        plt.title(title)
    plt.show()
    if output_filename:
        path = 'graphs/'
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + '{}.png'.format(output_filename), format="png")
        plt.show()

    return x_values, train_losses, validation_losses


def plot_loss_visualization_2d(base_model, data, build_model_function, mode='all', title=None, output_filename=None,
                               XYZ=None):
    """
    Visualizes the minimizer for a model along two random Gaussian filter-normalized directions.
    :param base_model: model to evaluate
    :param data: data to evaluate the model on, used to generate loss numbers
    :param mode: plotting mode.
       -'filled_contours': generate contours filled in with colors representing levels
       -'contours': generate contours with no fill
       -'surface': generate 3D surface plot
       -'all': generate all of the above
    :param title: title for the plot
    :param output_filename: file to save the plot to
    :param XYZ: tuple of (X, Y, Z) values. If provided, the function will skip loss computation and directly plot the values.
    :return: X, Y, Z
    """
    # Use XYZ parameter for plotting
    if XYZ:
        X, Y, Z = XYZ
    else:
        # Get weights and generate random directions
        weights = base_model.get_weights()
        direction_one = get_random_filter_normalized_direction(weights)
        direction_two = get_random_filter_normalized_direction(weights)

        # Set up new model and plotting variables
        x_values = np.linspace(-1, 1, 5)
        y_values = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.zeros((len(y_values), len(x_values)))
        new_model = build_model_function()

        # Compute loss for each linear combination of weight and direction
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                # Compute and set weights
                x = x_values[j]
                y = y_values[i]
                print("\n x: {}, y: {}".format(x, y))
                new_weights = [w + x * d1 + y * d2 for w, d1, d2 in zip(weights, direction_one, direction_two)]
                new_model.set_weights(new_weights)

                # Evaluate model
                loss, accuracy = new_model.evaluate(data)

                # Store losses
                Z[i, j] = loss

    # Plot results
    if mode == 'filled_contours':
        plt.contourf(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.colorbar()
    elif mode == 'contours':
        CS = plt.contour(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.clabel(CS, inline=1, fontsize=8)
    elif mode == 'surface':
        ax = plt.axes(projection='3d')
        ax.view_init(60, 35)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    elif mode == 'all':
        fig = plt.figure(figsize=(5, 15))
        # Plot filled contours
        ax1 = fig.add_subplot(3, 1, 1)
        cf = ax1.contourf(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.colorbar(cf, ax=ax1)

        # Plot contours
        ax2 = fig.add_subplot(3, 1, 2)
        cs = ax2.contour(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.clabel(cs, inline=1, fontsize=8)

        # Plot surface
        ax3 = fig.add_subplot(3, 1, 3, projection='3d')
        ax3.view_init(60, 35)
        ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # Add title and save plot if specified
    if title:
        plt.title(title)
    plt.show()
    if output_filename:
        path = 'graphs/'
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + '{}.png'.format(output_filename), format="png")
        plt.show()

    return X, Y, Z


def plot_mean_time_per_epoch(batch_sizes, mean_times, ylim=(0, 20)):
    rects = plt.bar(x=range(len(batch_sizes)), height=mean_times, tick_label=batch_sizes)
    plt.xlabel('Batch size')
    plt.ylabel('Average time per epoch (s)')
    plt.ylim(ylim)
    plt.title('Mean training time per epoch by batch size')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig('graphs/mean_epoch_training_time_by_batch_size.png', format="png")
    plt.show()


def histogram_num_of_train_epochs_until_conv(batch_sizes, convergence_epochs, ylim=(0, 1000)):
    rects = plt.bar(x=range(len(batch_sizes)), height=convergence_epochs, tick_label=batch_sizes)
    plt.xlabel('Batch size')
    plt.ylabel('Number of training epochs')
    plt.ylim(ylim)
    plt.title('Number of training epochs until convergence')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig('graphs/num_training_epochs_by_batch_size.png', format="png")
    plt.show()


def histogram_overall_time_until_end_of_epochs(batch_sizes, overall_training_times, ylim=(0, 2000)):
    rects = plt.bar(x=range(len(batch_sizes)), height=overall_training_times, tick_label=batch_sizes)
    plt.xlabel('Batch size')
    plt.ylabel('Overall training time (in seconds)')
    plt.ylim(ylim)
    plt.title('Overall training time until end of epochs, by batch size')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig('graphs/overall_training_time_by_batch_size.png', format="png")
    plt.show()


def histogram_sharpness(batch_sizes, sharpnesses):
    rects = plt.bar(x=range(len(batch_sizes)), height=sharpnesses, tick_label=batch_sizes)
    # autolabel(rects)
    plt.ylim(0, 105)
    plt.xlabel('Batch size')
    plt.ylabel('Sharpness')
    plt.title('Sharpness score by batch size')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig('graphs/sharpness_by_batch_size.png', format="png")
    plt.show()


def extract_times_for_batch_sizes(models_states, batch_sizes, key_tupel):
    mean_times = []
    convergence_epochs = []
    overall_training_times = []

    for batch_size in batch_sizes:
        # Compute mean epoch time, num epochs to converge, overall training time
        state = models_states[key_tupel][batch_size]
        mean_time_per_epoch = np.mean(state.times)
        convergence_epoch = np.argmin(state.history['val_loss'])
        overall_training_time = np.sum(state.times[:convergence_epoch])

        # Append results
        convergence_epochs.append(convergence_epoch)
        mean_times.append(mean_time_per_epoch)
        overall_training_times.append(overall_training_time)

        print("Batch size: ", batch_size)
        print("\tMean time per epoch: ", mean_time_per_epoch)
        print("\tConverged in {} epochs".format(convergence_epoch))
        print("\tOverall training time (in seconds) until convergence: ", overall_training_time)

    return mean_times, convergence_epochs, overall_training_times


def plot_sharpness(batch_sizes, sharpnesses, key, learning_rates=None):
    if learning_rates:
        sharpness_values = [sharpnesses[key + (batch_size, lr,)] for batch_size, lr in zip(batch_sizes, learning_rates)]
    else:
        sharpness_values = [sharpnesses[key + (batch_size,)] for batch_size in batch_sizes]
    ylim = (0, max(sharpness_values))

    plt.bar(x=range(len(batch_sizes)), height=sharpness_values, tick_label=batch_sizes)
    plt.ylim(ylim)
    plt.xlabel('Batch size')
    plt.ylabel('Sharpness')
    plt.title('Sharpness score by batch size')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig('graphs/sharpness_by_batch_size.png', format="png")
    plt.show()


def plot_distance_from_initial_weight(models, initial_weights, batch_sizes, key, learning_rates = None, ylim=(0, 14)):
    distances = []
    if learning_rates is None:
        for batch_size in batch_sizes:
            flattened_initial_weights = np.concatenate([w.flatten() for w in initial_weights[key + (batch_size,)]])
            model = models[key + (batch_size,)]
            flattened_weights = np.concatenate([w.flatten() for w in model.get_weights()])
            distance = np.linalg.norm(flattened_weights - flattened_initial_weights)
            distances.append(distance)
            print("Batch size: {}, distance: {}".format(batch_size, distance))

    else:
        for batch_size, lr in zip(batch_sizes, learning_rates):
            flattened_initial_weights = np.concatenate([w.flatten() for w in initial_weights[key + (batch_size, lr)]])
            model = models[key + (batch_size, lr)]
            flattened_weights = np.concatenate([w.flatten() for w in model.get_weights()])
            distance = np.linalg.norm(flattened_weights - flattened_initial_weights)
            distances.append(distance)
            print("Batch size: {}, distance: {}".format(batch_size, distance))

        

    # Plot distances
    plt.bar(x=range(len(batch_sizes)), height=distances, tick_label=batch_sizes)
    plt.xlabel('Batch size')
    plt.ylim()
    plt.ylabel('Distance from initial weights')
    plt.title('Distance from initial weights by batch size')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig('graphs/distance_from_initial_weights_by_batch_size.png', format="png")
    plt.show()


def plot_sharpness_times_runtime(batch_sizes, overall_training_times, sharpnesses, key, learning_rates=None):
    if learning_rates:
        sharpness_values = np.array(
            [sharpnesses[key + (batch_size, lr,)] for batch_size, lr in zip(batch_sizes, learning_rates)])
    else:
        sharpness_values = np.array([sharpnesses[key + (batch_size,)] for batch_size in batch_sizes])

    plt.bar(x=range(len(batch_sizes)), height=sharpness_values * np.array(overall_training_times),
            tick_label=batch_sizes)
    # plt.ylim(ylim)
    plt.xlabel('Batch size')
    plt.ylabel('Sharpness times runtime')
    plt.title('Sharpness score times runtime by batch size')
    path = 'graphs/'
    os.makedirs(path, exist_ok=True)
    plt.savefig('graphs/sharpness_times_runtime_by_batch_size.png', format="png")
    plt.show()
