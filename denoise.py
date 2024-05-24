from load_oxford_flowers102 import load_oxford_flowers102
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
import math
import os
import pickle, gzip
from sklearn.metrics import confusion_matrix


def createUNet(load_from_file = False, verbose = True, epoch = 50):
    #Load the data
    # data_train, data_validation, data_test, class_names = load_oxford_flowers102(imsize=32, fine=False)
    data_train, data_validation, data_test, class_names = load_oxford_flowers102(imsize=32, fine=True)
    x_train = data_train['images']
    x_train_normalized = x_train.astype(np.float32) / 255.0

    x_valid = data_validation['images']
    x_valid_normalized = x_valid.astype(np.float32) / 255.0

    x_test = data_test['images']

    x_test_normalized = x_test.astype(np.float32) / 255.0

    # Parameters for varying levels of noise
    # std_dev_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Adjust these values as needed
    std_dev_values = [0.1, 0.3, 0.5, 0.7]  # Adjust these values as needed
    # Initialize lists to store noisy images and their corresponding noisy counterparts
    noisy_images_train = []
    previous_noisy_images_train = []

    # Iterate over each image in x_train_normalized
    for image in x_train_normalized:
        # Apply Gaussian noise at each noise level and store the noisy images
        for i, std_dev in enumerate(std_dev_values):
            noisy_image = add_gaussian_noise(image, std_dev)
            noisy_images_train.append(noisy_image)
            noisy_image_prev= add_gaussian_noise(image, std_dev-.1)
            previous_noisy_images_train.append(noisy_image_prev)

    # Convert lists to NumPy arrays
    noisy_images_train = np.array(noisy_images_train)
    previous_noisy_images_train = np.array(previous_noisy_images_train)

    noisy_valid =[]
    previous_noisy_valid = []
    # Iterate over each image in the validation dataset
    for image in x_valid_normalized:
        # Select three random standard deviation values
        random_std_dev_values = np.random.choice(std_dev_values, 3, replace=False)
        # Apply Gaussian noise at each selected noise level and store the noisy images
        for std_dev in random_std_dev_values:
            noisy_image = add_gaussian_noise(image, std_dev)
            noisy_valid.append(noisy_image.copy())
            prev_noise = add_gaussian_noise(image, std_dev-.1)
            previous_noisy_valid.append(prev_noise)
            
    # Convert lists to NumPy arrays
    noisy_valid = np.array(noisy_valid)
    previous_noisy_valid = np.array(previous_noisy_valid)

    # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir('saved')

    save_name = os.path.join('saved', 'denoise%d' % (int)(epoch))
    net_save_name = save_name + '_cnn_net.h5'
    checkpoint_save_name = save_name + '_cnn_net.chk'
    history_save_name = save_name + '_cnn_net.hist'

    if load_from_file and os.path.isfile(net_save_name):
        # ***************************************************
        # * Loading previously trained neural network model *
        # ***************************************************

        # Load the model from file
        if verbose:
            print("Loading neural network from %s..." % net_save_name)
        model = tf.keras.models.load_model(net_save_name)
        model.summary()
        print("Total number of parameters:", model.count_params())

        # Load the training history - since it should have been created right after
        # saving the model
        if os.path.isfile(history_save_name):
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)
        else:
            history = []
    else:
        # ************************************************
        # * Creating and training a neural network model *
        # ************************************************
        # Define the input shape
        input_shape = (32, 32, 3)

        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Encoder
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bottleneck
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)

        # Decoder
        up6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
        up6 = tf.keras.layers.concatenate([up6, conv4], axis=3)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)

        up7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
        up7 = tf.keras.layers.concatenate([up7, conv3], axis=3)
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)

        up8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
        up8 = tf.keras.layers.concatenate([up8, conv2], axis=3)
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)

        up9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        up9 = tf.keras.layers.concatenate([up9, conv1], axis=3)
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)

        # Output layer
        outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='linear')(conv9)

        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        # model.compile(optimizer=optimizer,
        #             loss="mse",
        #             metrics="accuracy")

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="mse",
                  metrics="accuracy")
        
        # Print model summary
        # model.summary()

        print("Total number of parameters:", model.count_params())

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        
        train_info = model.fit(noisy_images_train, previous_noisy_images_train, validation_data=(noisy_valid, previous_noisy_valid),
                              epochs=epoch, shuffle=True, callbacks=[model_checkpoint_callback])
        
        # Load the weights of the best model
        print("Loading best save weight from %s..." % checkpoint_save_name)
        model.load_weights(checkpoint_save_name)

        # Save the entire model to file
        print("Saving neural network to %s..." % net_save_name)
        model.save(net_save_name)

        # Save training history to file
        history = train_info.history
        with gzip.open(history_save_name, 'w') as f:
            pickle.dump(history, f)


    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************
    # test the model agaisnt images with added noise
    plotNosieyTestImages(model, x_test_normalized)

    # generate an image from random noise
    generateFromNoise(model)
    generateFromNoise(model)

    

def generateFromNoise(model):
    # generaete images
    image_shape = 32 * 32
    std_dev = .5
    noisy_image = add_gaussian_noise_to_blank(image_shape, std_dev)

    num_iterations = 25

    denoised_images = []
    denoised_images.append(noisy_image)

    # Iterate and denoise the image
    for i in range(num_iterations):  
        # Pass the noisy image through the denoising model
        denoised_image = model.predict(np.expand_dims(noisy_image, axis=0))[0]
        
        # Update the noisy image with the denoised version for the next iteration
        noisy_image = denoised_image
        
        # Add the denoised image to the list
        denoised_images.append(denoised_image)

    # Plot the denoised images
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(denoised_images):
        plt.subplot(1, len(denoised_images), i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Create subplots with 1 row and 2 columns
    plt.figure(figsize=(12, 5))

    # Calculate the index for the halfway image
    halfway_index = math.ceil(len(denoised_images) / 2)  # Round up to the nearest whole number

    # Plot the halfway image
    plt.subplot(1, 2, 1)
    plt.imshow(denoised_images[halfway_index])
    plt.title('Halfway Image')
    plt.axis('off')

    # Plot the last denoised image
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_images[-1])
    plt.title('Last Denoised Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plotNosieyTestImages(model, data):
    noisy_images_test = []

    # Iterate over each image in x_train_normalized
    for image in data:
        # Apply Gaussian noise at each noise level and store the noisy images
        noisy_image = add_gaussian_noise(image, 0.5)
        noisy_images_test.append(noisy_image)

        # Convert lists to NumPy arrays
    noisy_images_test = np.array(noisy_images_test)

    # Predict denoised images for the noisy test images
    predicted_images = model.predict(noisy_images_test)
    # Plot the noisy test images and their corresponding denoised images
    plt.figure(figsize=(12, 6))
    for i in range(4):
        # Plot the original image
        plt.subplot(3, 4, i + 1)
        plt.imshow(data[i])
        plt.title('Original Image {}'.format(i + 1))
        plt.axis('off')

        # Plot the noisy test image
        plt.subplot(3, 4, i + 5)
        plt.imshow(noisy_images_test[i])
        plt.title('Noisy Image {}'.format(i + 1))
        plt.axis('off')

        # Plot the corresponding denoised image
        plt.subplot(3, 4, i + 9)
        plt.imshow(predicted_images[i])
        plt.title('Predicted Image {}'.format(i + 1))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def add_gaussian_noise(image, std_dev):
    noise = np.random.normal(loc=0, scale=std_dev, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)  # Clip pixel values to [0, 1]

def add_gaussian_noise_to_blank(shape, std_dev):
    random_noise = np.random.rand(32, 32, 3) * 255  # Scale to [0, 255]

    # Clip the values to ensure they are in the valid range [0, 255]
    random_noise = np.clip(random_noise, 0, 255)

    # Normalize pixel values to float32
    random_noise = random_noise.astype(np.float32) / 255.0
    
    return random_noise


if __name__ == "__main__":
    createUNet(load_from_file=True, verbose=True, epoch = 20)