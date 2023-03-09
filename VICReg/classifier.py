import tensorflow as tf

class ClusterClassifier(tf.keras.Model):
    def __init__(self, encoder, classifier):
        super(ClusterClassifier, self).__init__()
        self.encoder    = encoder
        self.classifier = classifier
    
    def train_step(self,data):
        x, y, sample_weight = data
        
        z = self.encoder(x, training=False)

        with tf.GradientTape() as tape:
            y_pred = self.classifier(z, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.classifier.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, data):
        data = self.encoder(data, training=False)
        data = self.classifier(data, training=False)
        return data
    
#     def test_step(self, data):
#         x, y = data
#         y_pred = self.encoder(x, training=False)
#         z_pred = self.classifier(y_pred, training=False)
#         self.compiled_loss(y, z_pred, regularization_losses=self.losses)
#         # Update the metrics.
#         self.compiled_metrics.update_state(y, z_pred)
#         # Return a dict mapping metric names to current value.
#         # Note that it will include the loss (tracked in self.metrics).
#         return {m.name: m.result() for m in self.metrics}

class classifier_class(tf.keras.Model):
    def __init__(self, input_shape, output_size, num_layers=1, layer_widths=[256], final_activation='softmax'):
        super(classifier_class, self).__init__()
        self.input_layer = tf.keras.layers.Input(input_shape)
        self.layer_list = [tf.keras.layers.Flatten(),tf.keras.layers.BatchNormalization()]
        for i in range(num_layers):
            self.layer_list.append(tf.keras.layers.Dense(layer_widths[i], activation='relu'))
            self.layer_list.append(tf.keras.layers.BatchNormalization())
        self.layer_list.append(tf.keras.layers.Dense(output_size, activation=final_activation))
        self.out = self.call(self.input_layer)
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            for layer in self.layer_list:
                x = layer(x, training=True)
            loss = self.compiled_loss(y, x, regularization_losses=self.losses)
        
        # Compute gradients
        trainable_vars = self.classifier.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, x)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, x, **kwargs):
        for layer in self.layer_list:
            x = layer(x, **kwargs)
        return x
