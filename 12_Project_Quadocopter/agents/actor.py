from keras import backend as K
from keras import layers, models, optimizers

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high, hidden_sizes=(512,256), L2=0.01):
        self.state_size   = state_size
        self.action_size  = action_size
        self.action_low   = action_low
        self.action_high  = action_high
        self.action_range = action_high - action_low
        self.build_model(hidden_sizes, L2)

    def build_model(self, hidden_sizes, L2):
        states = layers.Input(shape=(self.state_size,), name='states')

        net = layers.Dense(units=hidden_sizes[0], kernel_regularizer=layers.regularizers.l2(L2))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=hidden_sizes[1], kernel_regularizer=layers.regularizers.l2(L2))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net) 
        # output => from 0 to 1
        out = layers.Dense(units=self.action_size, activation='sigmoid', name='out', kernel_initializer=layers.initializers.RandomUniform(minval=-0.03, maxval=0.03))(net)
        # actions => from action_low to action_high
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(out)
        # set input & output
        self.model = models.Model(inputs=states, outputs=actions)

        # calculate loss
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        optimizer = optimizers.Adam(lr=L2)
        updates_op = optimizer.get_updates(loss, self.model.trainable_weights)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates_op)