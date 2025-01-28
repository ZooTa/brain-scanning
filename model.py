import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import random

class RetinaModel:
    def __init__(self):
        self.model = self.build_model()
        self.BATCH_SIZE = 16  
        self.train_directory = ''
        self.test_directory = ''
        
        
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', input_shape=(192, 192, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    
    def class_distribution(self, dataset):
        class_values = []
        total_batches = dataset.__len__().numpy()
        for batch, element in enumerate(dataset.as_numpy_iterator()):
            if batch+1 == total_batches:
                for i in range(len(element[1])):
                    class_values.append(self.class_names[int(element[1][i])])
            else:
                for i in range(self.BATCH_SIZE):
                    class_values.append(self.class_names[int(element[1][i])])

        class_n, frequency = np.unique(np.array(class_values), return_counts=True)
        return pd.DataFrame(frequency, class_n, columns=["Count"])
    
    
    
    def convert_to_float(self, image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float16)
        return image, label
    
    
    def preprocessing(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        
        self.train = tf.keras.preprocessing.image_dataset_from_directory(
        self.train_directory,
        labels='inferred',
        seed=0,
        image_size=(192, 192),
        batch_size=self.BATCH_SIZE,
        color_mode='rgb',
        validation_split=0.2,
        subset='training'
        )

        self.validation = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_directory,
            labels='inferred',
            seed=0,
            image_size=(192, 192),
            batch_size=self.BATCH_SIZE,
            color_mode='rgb',
            validation_split=0.2,
            subset='validation'

        )

        self.test = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_directory,
            labels='inferred',
            image_size=(192, 192),
            batch_size=self.BATCH_SIZE,
            color_mode='rgb'

        )
        test_labels = [labels for _, labels in self.test.unbatch()]
        self.class_names=self.train.class_names
        train_class_dist = self.class_distribution(self.train)
        total = len(self.train.file_paths)
        count_cnv = 29709
        count_dme = 9113
        count_drusen = 6917
        count_normal = 21049
        cnv_weight = (1/count_cnv) * (total/4)
        dme_weight = (1/count_dme) * (total/4)
        drusen_weight = (1/count_drusen) * (total/4)
        norm_weight = (1/count_normal) * (total/4)
        class_weight = {0 : cnv_weight, 1: dme_weight, 2 : drusen_weight, 3: norm_weight}
        


    def train(self, train_data, epochs, checkpoint_dir):
        
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train = (
            self.train
            .map(self.convert_to_float)
            .cache()
            .prefetch(buffer_size=AUTOTUNE)
        )
        validation = (
            self.validation
            .map(self.convert_to_float)
            .cache()
            .prefetch(buffer_size=AUTOTUNE)
        )
        test = (
            self.test
            .map(self.convert_to_float)
            .cache()
            .prefetch(buffer_size=AUTOTUNE)
        )
        
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            save_weights_only=True,
        )
        
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            train,
            validation_data=validation,
            epochs=epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
            shuffle=True
        )

        return history

    def save_model(self):
        self.model.save("my3_model.h5")

    def load_weights(self, checkpoint_dir):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            self.model.load_weights(latest_checkpoint)
            return True
        return False

    def predict(self, image):
        # Remove the argument from preprocessing method
        self.preprocessing()  # Call the preprocessing method
        image = np.expand_dims(image, axis=0)  # Add a batch dimension
        prediction = self.model.predict(image)
        return prediction