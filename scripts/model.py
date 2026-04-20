import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="scripts")
class FaceEmbedder(tf.keras.Model):
    def __init__(
        self,
        input_shape=(160, 160, 3),
        embedding_dim=128,
        base_filters=32,
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(input_shape)
        self.embedding_dim = int(embedding_dim)
        self.base_filters = int(base_filters)
        self.dropout_rate = float(dropout_rate)

        self.rescale = tf.keras.layers.Rescaling(1.0 / 255.0)

        self.conv1 = tf.keras.layers.Conv2D(self.base_filters, 3, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(self.base_filters * 2, 3, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(self.base_filters * 4, 3, padding="same", use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(self.base_filters * 4, 3, padding="same", use_bias=False)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.proj = tf.keras.layers.Dense(self.embedding_dim, use_bias=True)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, images, training=False):
        x = self.rescale(images)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.act(x)
        x = self.gap(x)

        x = self.dropout(x, training=training)
        x = self.proj(x)
        x = self.ln(x, training=training)

        return tf.nn.l2_normalize(x, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_,
            "embedding_dim": self.embedding_dim,
            "base_filters": self.base_filters,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.saving.register_keras_serializable(package="scripts")
class FaceNetEmbedder(tf.keras.Model):
    def __init__(
        self,
        input_shape=(160, 160, 3),
        embedding_dim=128,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(input_shape)
        self.embedding_dim = int(embedding_dim)
        self.trainable_ = bool(trainable)

        self.resize = tf.keras.layers.Resizing(self.input_shape_[0], self.input_shape_[1])

        self._facenet = None

        self.proj = None
        if self.embedding_dim != 512:
            self.proj = tf.keras.layers.Dense(self.embedding_dim, use_bias=True, name="proj")

    def _ensure_loaded(self):
        if self._facenet is not None:
            return
        from keras_facenet import FaceNet

        self._facenet = FaceNet()

    def _embeddings_np(self, images_np):
        self._ensure_loaded()
        try:
            return self._facenet.embeddings(images_np, verbose=0)
        except TypeError:
            return self._facenet.embeddings(images_np)

    def call(self, images, training=False):
        x = tf.cast(images, tf.float32)
        x = self.resize(x)

        def _fn(imgs):
            return self._embeddings_np(imgs)

        emb = tf.numpy_function(_fn, [x], tf.float32)
        emb = tf.ensure_shape(emb, [None, 512])

        if self.proj is not None:
            emb = self.proj(emb)

        return tf.nn.l2_normalize(emb, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_,
            "embedding_dim": self.embedding_dim,
            "trainable": self.trainable_,
        })
        return config


@keras.saving.register_keras_serializable(package="scripts")
class SiameseVerifier(tf.keras.Model):
    def __init__(
        self,
        input_shape=(160, 160, 3),
        embedding_dim=128,
        base_filters=32,
        dropout_rate=0.2,
        embedder_type: str = "cnn",
        embedder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(input_shape)
        self.embedding_dim = int(embedding_dim)
        self.base_filters = int(base_filters)
        self.dropout_rate = float(dropout_rate)

        self.embedder_type = str(embedder_type)

        if embedder is not None:
            self.embedder = embedder
        elif self.embedder_type == "cnn":
            self.embedder = FaceEmbedder(
                input_shape=self.input_shape_,
                embedding_dim=self.embedding_dim,
                base_filters=self.base_filters,
                dropout_rate=self.dropout_rate,
            )
        elif self.embedder_type == "facenet":
            self.embedder = FaceNetEmbedder(
                input_shape=self.input_shape_,
                embedding_dim=self.embedding_dim,
                trainable=False,
            )
        else:
            raise ValueError(
                "Unknown embedder_type for SiameseVerifier: "
                f"{self.embedder_type!r}. Expected 'cnn' or 'facenet'."
            )
        self.logit_scale = tf.keras.layers.Dense(1, use_bias=True, name="logit_scale")

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.BinaryAccuracy(
            name="accuracy",
            threshold=0.0,
        )
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_,
            "embedding_dim": self.embedding_dim,
            "base_filters": self.base_filters,
            "dropout_rate": self.dropout_rate,
            "embedder_type": self.embedder_type,
            "embedder": None,
        })
        return config    

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def encode(self, images, training=False):
        return self.embedder(images, training=training)

    def call(self, inputs, training=False):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("SiameseVerifier.call expects (img_a, img_b)")
        img_a, img_b = inputs

        z_a = self.encode(img_a, training=training)
        z_b = self.encode(img_b, training=training)

        cos_sim = tf.reduce_sum(z_a * z_b, axis=-1, keepdims=True)
        logits = self.logit_scale(cos_sim)
        return logits

    def predict_proba(self, img_a, img_b, training=False):
        logits = self((img_a, img_b), training=training)
        return tf.sigmoid(logits)

    def train_step(self, data):
        (img_a, img_b), y = data
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, (-1, 1))

        with tf.GradientTape() as tape:
            logits = self((img_a, img_b), training=True)
            loss = self.loss_fn(y, logits)
            loss += tf.add_n(self.losses) if self.losses else 0.0

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (img_a, img_b), y = data
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, (-1, 1))

        logits = self((img_a, img_b), training=False)
        loss = self.loss_fn(y, logits)
        loss += tf.add_n(self.losses) if self.losses else 0.0

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

