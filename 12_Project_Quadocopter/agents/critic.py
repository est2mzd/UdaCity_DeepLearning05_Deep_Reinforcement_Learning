from keras import backend as K
from keras import layers, models, optimizers

class Critic:
    def __init__(self, state_size, action_size, hidden_sizes=(512,256,256), L2=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model(hidden_sizes,L2)

    def build_model(self,hidden_sizes,L2):
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        net = layers.Dense(units=hidden_sizes[0], kernel_regularizer=layers.regularizers.l2(L2))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=hidden_sizes[1], kernel_regularizer=layers.regularizers.l2(L2))(net)

        net_actions = layers.Dense(units=hidden_sizes[2],kernel_regularizer=layers.regularizers.l2(L2))(actions)

        net = layers.Add()([net, net_actions])
        net = layers.Activation('relu')(net)

        out = layers.Dense(units=1, name='q_values', kernel_initializer=layers.initializers.RandomUniform(minval=-0.3, maxval=0.3))(net)

        self.model = models.Model(inputs=[states, actions], outputs=out)
        self.model.compile(optimizer=optimizers.Adam(lr=L2), loss='mse')
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=K.gradients(out, actions))