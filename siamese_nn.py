import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, metrics, Model, applications
# Local Imports
import model_references as MR

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @tf.function
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin:float=0.5, output_size:int=64):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.output_size = output_size
    
    @tf.function
    def call(self, inputs):
        return self.siamese_network(inputs)
    
    @tf.function
    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    @tf.function
    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function
    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]        

    def create_model(self, target_shape:tuple=(224,224), model:str='RESNET50V2', weights:str='IMAGENET'): 
        base_cnn = MR.MODEL[model](weights=MR.WEIGHTS[weights], input_shape=target_shape + (3,), include_top=False)

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense3 = layers.Dense(128, activation="relu")(dense2) # Added
        dense3 = layers.BatchNormalization()(dense3) # Added
        output = layers.Dense(self.output_size)(dense2) # was 256

        self.embedding_model = Model(base_cnn.input, output, name="Embedding")

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
        positive_input = layers.Input(name="positive", shape=target_shape + (3,))
        negative_input = layers.Input(name="negative", shape=target_shape + (3,))
        parent_model = MR.PREPROCESS[model]

        distances = DistanceLayer()(
            self.embedding_model(parent_model.preprocess_input(anchor_input)),
            self.embedding_model(parent_model.preprocess_input(positive_input)),
            self.embedding_model(parent_model.preprocess_input(negative_input)),
        )

        siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

        self.siamese_model = SiameseModel(siamese_network,1)
        self.siamese_model.compile(optimizer=optimizers.Adam(0.0001), weighted_metrics=[])
    
    