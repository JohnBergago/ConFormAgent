import numpy as np
from typing import Callable, Dict, List, Optional
import tensorflow as tf

ActivationFunction = Callable[[tf.Tensor], tf.Tensor]


def get_conv_out_size(width, kernel, stride, padding):
    return int((width-kernel+2*padding)/stride)+1


def get_cnn_out_dimension(
    input_dimensions, 
    cnn_config : List[List[int]]
    ):
    """Calculates the output dimensions of a CNN based on the input dimensions 
    and layer configuration.

    Args:
        input_dimensions: Shape of the input for the convolutional 
            layers. [x, y, channels]
        cnn_config (List[List[int]]): List specifiying the convolutional layers. 
            Each layer is specified by a list of 3 properties:
            (filters, kernel_size, stride)

    Returns:
        List[int]: Output dimensions as [x, y, channels]
    """

    out_dim = input_dimensions
    for layer_spec in cnn_config:
        dim_x = get_conv_out_size(out_dim[0], layer_spec[1], layer_spec[2], 0)
        dim_y = get_conv_out_size(out_dim[1], layer_spec[1], layer_spec[2], 0)
        out_dim = [dim_x, dim_y, layer_spec[0]]
    return out_dim


def swish(input_activation: tf.Tensor) -> tf.Tensor:
    """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
    return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))


def create_ffn(
        observation_input,
        activation: ActivationFunction,
        dense_layers: List[int],
        kernel_initializer,
        flatten: bool = False,
):
    """ Builds a set of hidden state encoders.

    Args:
        observation_input: Input tensor.
        activation (ActivationFunction): What type of activation function to 
            use for layers.
        dense_layers (List[int]): 1-D list with each element representing the size 
            of a hidden layer
        kernel_initializer: Initialization function for the kernel.
        flatten (bool, optional): If true, the output will be flattened into one
            dimension. Defaults to False.

    Returns:
        List[tf.Tensor]: List of hidden layer tensors.
    """
    hidden = observation_input
    dense_layer_count = 1
    for num_units in dense_layers:
        hidden = tf.keras.layers.Dense(
            num_units,
            activation=activation,
            name="ffn_{}".format(dense_layer_count),
            kernel_initializer=kernel_initializer, )(
            hidden)
        dense_layer_count += 1
    if flatten:
        hidden = tf.keras.layers.Flatten()(hidden)
    return hidden


def create_cnn(
        observation_input,
        activation: ActivationFunction,
        conv_layers,
        flatten: bool = False,
):
    """Builds a set of convolutional layers.

    Args:
        observation_input: Input tensor.
        activation (ActivationFunction): What type of activation function to use for layers.
        conv_layers: List specifiying the convolutional layer. Each layer 
            is specified by a list of 3 properties (filters, kernel_size, stride)
        flatten (bool, optional): If true, the output will be flattened into one dimension. Defaults to False.

    Returns:
       : List of hidden layer tensors.
    """

    hidden = observation_input
    conv_layer_count = 1
    for layer_spec in conv_layers:
        hidden = tf.keras.layers.Conv2D(
            filters=layer_spec[0],
            kernel_size=layer_spec[1],
            strides=layer_spec[2],
            name="conv_{}".format(conv_layer_count),
            activation=activation, )(
            hidden)
        conv_layer_count += 1

    if flatten:
        hidden = tf.keras.layers.Flatten()(hidden)
    return hidden