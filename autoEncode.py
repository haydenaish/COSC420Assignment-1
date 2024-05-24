from load_oxford_flowers102 import load_oxford_flowers102
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
import os
import pickle, gzip
from sklearn.metrics import confusion_matrix


def createUNet(load_from_file = False, verbose = True, eval = False, epoch = 50):
    #Load the data
    data_train, data_validation, data_test, class_names = load_oxford_flowers102(imsize=32, fine=False)
    x_train = data_train['images']
    x_train_normalized = x_train.astype(np.float32) / 255.0
    # x_train_normalized = tf.cast(x_train, tf.float32)/255

    x_valid = data_validation['images']
    # x_valid_normalized = tf.cast(x_valid, tf.float32)/255
    x_valid_normalized = x_valid.astype(np.float32) / 255.0

    x_test = data_test['images']

    # x_test_normalized = tf.cast(x_test, tf.float32)/255
    x_test_normalized = x_test.astype(np.float32) / 255.0

    # x_train_normalized = np.reshape(x_train_normalized, (data_train['images'].shape[0], data_train['images'].shape[1], data_train['images'].shape[2], 3))
    # x_valid_normalized = np.reshape(x_valid_normalized, (data_validation['images'].shape[0], data_validation['images'].shape[1], data_validation['images'].shape[2], 3))
    # x_test_normalized = np.reshape(x_test_normalized, (data_test['images'].shape[0], data_test['images'].shape[1], data_test['images'].shape[2],3))


    # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir('saved')

    save_name = os.path.join('saved', 'unet_ep%d' % (int)(epoch))
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

        # Bottleneck with reduced params
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
        
        train_info = model.fit(x_train_normalized, x_train_normalized, validation_data=(x_valid_normalized, x_valid_normalized),
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

    # Plot training and validation accuracy over the course of training
    if verbose and history != []:
        fh = plt.figure()
        ph = fh.add_subplot(111)
        ph.plot(history['accuracy'], label='accuracy')
        ph.plot(history['val_accuracy'], label = 'val_accuracy')
        ph.set_xlabel('Epoch')
        ph.set_ylabel('Accuracy')
        ph.set_ylim([0, 1])
        ph.legend(loc='lower right')

    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************
    if verbose and eval:
        reconstructed_images = model.predict(x_test_normalized)
        # Compute pixel-wise error
        reconstruction_errors = np.abs(x_test_normalized - reconstructed_images)

        # Compute mean and standard deviation of error
        mean_error = np.mean(reconstruction_errors)
        std_dev_error = np.std(reconstruction_errors)
        print("Mean Pixel-wise Error:", mean_error)
        print("Standard Deviation of Pixel-wise Error:", std_dev_error)

        # Create table data
        table_data = [
            ['Metric', 'Value'],
            ['Mean Pixel-wise Error', f'{mean_error:.4f}'],
            ['Standard Deviation of Pixel-wise Error', f'{std_dev_error:.4f}']
        ]

        # Plot the table
        plt.figure(figsize=(6, 4))
        plt.table(cellText=table_data,
                colWidths=[0.6, 0.4],
                cellLoc='center',
                loc='center')
        plt.axis('off')
        plt.title('Reconstruction Error Metrics')
        plt.show()

        # Randomly select 5 indices
        random_indices = np.random.choice(len(reconstruction_errors), 5, replace=False)

        # Visualize the randomly selected reconstructed images
        plt.figure(figsize=(18, 9))  # Adjust figure size accordingly
        for i, idx in enumerate(random_indices):
            input_image = x_test_normalized[idx]
            reconstructed_image = reconstructed_images[idx]

            # Plot the input image
            plt.subplot(2, 5, i + 1)  # Use 2 rows and 5 columns for 5 images
            plt.imshow(input_image)
            plt.title(f'Input Image {i+1}')
            plt.axis('off')

            # Plot the reconstructed predicted image
            plt.subplot(2, 5, i + 6)  # Second row starts at index 6
            plt.imshow(reconstructed_image)
            plt.title(f'Reconstructed Predicted Image {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # # Sort the reconstruction errors
        sorted_indices = np.argsort(np.mean(reconstruction_errors, axis=(1, 2, 3)))

        # # Get the indices of the three best and three worst reconstructed images
        best_indices = sorted_indices[:3]
        worst_indices = sorted_indices[-3:]

        # Visualize the three best reconstructed images
        plt.figure(figsize=(12, 6))
        for i, idx in enumerate(best_indices):
            input_image = x_test_normalized[idx]
            reconstructed_image = reconstructed_images[idx]

            # Plot the input image
            plt.subplot(2, 3, i + 1)
            plt.imshow(input_image)
            plt.title(f'Input Image {i+1}')
            plt.axis('off')

            # Plot the reconstructed predicted image
            plt.subplot(2, 3, i + 4)
            plt.imshow(reconstructed_image)
            plt.title(f'Reconstructed Predicted Image {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        for i, idx in enumerate(worst_indices):
            input_image = x_test_normalized[idx]
            reconstructed_image = reconstructed_images[idx]

            # Plot the input image
            plt.subplot(2, 3, i + 1)
            plt.imshow(input_image)
            plt.title(f'Input Image {i+1}')
            plt.axis('off')

            # Plot the reconstructed predicted image
            plt.subplot(2, 3, i + 4)
            plt.imshow(reconstructed_image)
            plt.title(f'Reconstructed Predicted Image {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    


if __name__ == "__main__":
    # Load the submitted model with line below
    createUNet(load_from_file=True, verbose=True, eval = True, epoch = 101)