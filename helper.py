from ast import main
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint







def helper(model, train_generator, test_generator, early_stopping, model_checkpoint):
    m = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=50,
        validation_data=test_generator,
        validation_steps=test_generator.n // test_generator.batch_size,
        callbacks=[early_stopping, model_checkpoint],
        use_multiprocessing=True)
    return m