import numpy as np
from typing import Callable, Dict, List, Optional
import tensorflow as tf
from tensorflow.keras import layers
from ray.rllib.policy.rnn_sequencing import add_time_dimension

from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.tune.utils import deep_update
from conform_agent.models.tf.utils import swish, create_ffn, create_cnn

DEFAULT_MODEL_CONFIG={
    # Defines the convolutiontional layers. For each layer there has to be 
    # [num_filters, kernel, stride]. 
    "conv_layers": [
        [16, [8, 8], 4], 
        [16, [4, 4], 2]],
    # Defines the dense layers following the convolutional layers (if any). 
    # For each layer the num_hidden units has to be defined. 
    "dense_layers": [128, 128],
    # whether to use a LSTM layer after the dense layers.
    "use_recurrent": False,
    # The number of LSTM cells to use.
    "cell_size": 128,
}

class SimpleRCNNModel(RecurrentNetwork):

    def __init__(self, 
                obs_space, 
                action_space, 
                num_outputs, 
                model_config,
                name,
                **kw):
        super(SimpleRCNNModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        self.activation_fn = swish
        
        # Check if the configuration is valid
        custom_config = model_config.get("custom_model_config")
        if custom_config:
            custom_config = deep_update(DEFAULT_MODEL_CONFIG,
                                custom_config, 
                                new_keys_allowed=False, 
                                allow_new_subkey_list=[])
        else:
            custom_config = DEFAULT_MODEL_CONFIG
        
        # gather values from config if exists.
       
        if (len(np.asarray(custom_config.get("dense_layers")).shape) < 2):
            dense_layers = custom_config.get("dense_layers")
        else:
            raise Exception("Wrong shape for dense_layers options. Needs to be "
                "a 1-D list" )

        conv_layers:List[List[int]] = custom_config.get("conv_layers")
        self.use_recurrent = custom_config.get("use_recurrent")
        self.cell_size = custom_config.get("cell_size")
        
        # Define the core model layers which will be used by the other
        if self.use_recurrent:
            input_layer = tf.keras.layers.Input(
                shape=(None, ) + obs_space.shape,
                name="observations")
            state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ), name="h")
            state_in_c = tf.keras.layers.Input(shape=(self.cell_size, ), name="c")
            seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)
        else:
            input_layer = tf.keras.layers.Input(
                shape=obs_space.shape, 
                name="observations")

        # is it a visual input?
        if (len(obs_space.shape) > 1):           
            input_visual = tf.reshape(
                input_layer, (-1,) + obs_space.shape)
            cnn = create_cnn(
                input_visual,
                activation=tf.nn.elu,
                conv_layers=conv_layers,
                flatten=True)
            hidden = create_ffn(
                cnn,
                activation=self.activation_fn,
                dense_layers=dense_layers,
                kernel_initializer=tf.initializers.VarianceScaling(1.0))
        else:
            hidden = tf.reshape(
                input_layer, (-1,) + obs_space.shape)
            hidden = create_ffn(
                hidden,
                activation=self.activation_fn,
                dense_layers=dense_layers,
                kernel_initializer=tf.initializers.VarianceScaling(1.0))

        if self.use_recurrent:
            hidden = tf.reshape(
                hidden, [-1, tf.shape(input_layer)[1], hidden.shape.as_list()[-1]])
            lstm_out, state_h, state_c = tf.keras.layers.LSTM(
                self.cell_size, 
                return_sequences=True, 
                return_state=True, 
                name="lstm")(
                    inputs=hidden,
                    mask=tf.sequence_mask(seq_in),
                    initial_state=[state_in_h, state_in_c])
            hidden = lstm_out

        # Postprocess the networks output
        policy_layer = tf.keras.layers.Dense(
            num_outputs,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.initializers.VarianceScaling(0.01),
            name="policy",
        )(hidden)

        value_layer = tf.keras.layers.Dense(
            1, 
            name="value",
        )(hidden)

        if self.use_recurrent:
            self.base_model = tf.keras.Model(
                inputs=[input_layer, seq_in, state_in_h, state_in_c],
                outputs=[policy_layer, value_layer, state_h, state_c])
        else:
            self.base_model = tf.keras.Model(
                inputs=input_layer, 
                outputs=[policy_layer, value_layer])
        self.register_variables(self.base_model.variables)

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        if self.use_recurrent:
            return super(SimpleRCNNModel, self).forward(input_dict, state, seq_lens)
        else:
            model_out, self._value_out = self.base_model(input_dict["obs"])
            return model_out, state

    def forward_rnn(self, inputs, state, seq_lens):
        if len(self.obs_space.shape) > 1:
            inputs = tf.reshape(
                inputs, 
                shape=[
                    tf.shape(inputs)[0], 
                    tf.shape(inputs)[1],
                    self.obs_space.shape[0],
                    self.obs_space.shape[1],
                    self.obs_space.shape[2]],
                name="reshape_input")
        model_out, self._value_out, h, c = self.base_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    def get_initial_state(self):
        if self.use_recurrent:
            return [
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
            ]
        else:
            return []

    def value_function(self):
        return tf.reshape(self._value_out, [-1])