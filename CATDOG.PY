from ast import main
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('cat_dog_model.h5', save_best_only=True)

# Set number of workers
num_workers = 10


def setup():
    # Prepare the dataset
    train_dir = 'train/'
    test_dir = 'test/'

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Train the model
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                metrics=['accuracy'])
    return model, train_generator, test_generator



def evaluate(model, test_generator):
    # Evaluate the model
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
    print('Test accuracy:', test_acc)

# Make predictions on new images
# import numpy as np
# from keras.preprocessing import image

# img_path = 'path/to/new/image.jpg'
# img = image.load_img(img_path, target_size=(150, 150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.

# prediction = model.predict(img_tensor)

# if prediction < 0.5:
#     print('This is a cat.')
# else:
#     print('This is a dog.')

if __name__ == '__main__':
    # model, train_generator, test_generator = setup()
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=train_generator.n // train_generator.batch_size,
    #     epochs=50,
    #     validation_data=test_generator,
    #     validation_steps=test_generator.n // test_generator.batch_size,
    #     callbacks=[early_stopping, model_checkpoint],
    #     use_multiprocessing=True,
    #     workers=num_workers)
    # evaluate(model, test_generator)
    
    print(tf.config.list_physical_devices('GPU'))
