from load_oxford_flowers102 import load_oxford_flowers102
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
import os
import pickle, gzip
from sklearn.metrics import confusion_matrix
import pandas as pd

def ex1(load_from_file = False, verbose = True, opt ='adam', epoch = 50, confus = False, fine = False,
              reg_wdecay_beta = 0, reg_dropout_rate = 0, reg_batch_norm = False, data_aug=False):

   # Load the oxford flowers dataset
   data_train, data_validation, data_test, class_names = load_oxford_flowers102(imsize=32, fine=fine)
   # data_train, data_validation, data_test, class_names = load_oxford_flowers102(imsize=32, fine=fine)
   x_train = data_train['images']
   y_train = data_train['labels']
   # x_train_normalized = x_train.astype(np.float32) / 255.0
   x_train_normalized = tf.cast(x_train, tf.float32)/255

   x_valid = data_validation['images']
   y_valid = data_validation['labels']
   x_valid_normalized = x_valid.astype(np.float32) / 255.0

   x_test = data_test['images']
   y_test = data_test['labels']
   x_test_normalized = x_test.astype(np.float32) / 255.0
   n_classes = len(class_names)
   # print(opt)
   # Calculate class distribution for training set
   train_class_distribution = np.bincount(y_train) / len(y_train) * 100

   # # Compute class weights inversely proportional to class frequencies to use as class weights
   # class_weights = 1.0 / train_class_distribution

   # # Normalize the class weights to sum up to the number of classes
   # class_weights_normalized = class_weights / np.sum(class_weights)
   
   # class_weights_normalized +=1


   # # Create a dictionary mapping class indices to class weights
   # class_weights_dict = {i: class_weights_normalized[i] for i in range(len(class_weights_normalized))}


   # Print the computed class weights
   # print("Class Weights:", class_weights_normalized)


   # Data augmentation does not like shape (N,1) for labels, it must
   # be shape (N,)...and the squeeze function removes dimensions of size 1
   y_train = np.squeeze(y_train)
   y_test = np.squeeze(y_test)

   # Create 'saved' folder if it doesn't exist
   if not os.path.isdir("saved"):
      os.mkdir('saved')

   # Specify the names of the save files
   # save_name = os.path.join('saved', 'cifar10_rwd%.1e_rdp%.1f_rbn%d_daug%d_opt%s_epc%d' % (reg_wdecay_beta,reg_dropout_rate,int(reg_batch_norm),int(data_aug), opt, int(epoch)))
   save_name = os.path.join('saved', 'model%.1e_rdp%.1f_rbn%d_daug%d_opt%s_epc%d' % (reg_wdecay_beta,reg_dropout_rate,int(reg_batch_norm),int(data_aug), opt, int(epoch)))
   net_save_name = save_name + '_cnn_net.h5'
   checkpoint_save_name = save_name + '_cnn_net.chk'
   history_save_name = save_name + '_cnn_net.hist'
#    # Show 16 train images with the corresponding labels
   if verbose:
      x_train_sample = x_train[:16]
      y_train_sample = y_train[:16]
      show_methods.show_data_images(images=x_train_sample,labels=y_train_sample,class_names=class_names,blocking=False)

   n_classes = len(class_names)

   if load_from_file and os.path.isfile(net_save_name):
      # ***************************************************
      # * Loading previously trained neural network model *
      # ***************************************************

      # Load the model from file
      if verbose:
         print("Loading neural network from %s..." % net_save_name)
      net = tf.keras.models.load_model(net_save_name)

      # net.summary()

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

      # Create feed-forward network
      net = tf.keras.models.Sequential()

      # Conv layer 1: 3x3 window, 64 filters - specify the size of the input as 32x32x3
      net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                                     input_shape=(32, 32, 3)))

      # Conv layer 2: 3x3 window, 64 filters
      net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Conv layer 3: 3x3 window, 64 filters
      net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      if reg_batch_norm:
         # Batch norm layer 1
         net.add(tf.keras.layers.BatchNormalization())

      # Max pool layer 1: 2x2 window (implicit arguments - padding="valid")
      net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

      # Conv layer 4: 3x3 window, 128 filters
      net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Conv layer 5: 3x3 window, 128 filters
      net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Conv layer 6: 3x3 window, 128 filters
      net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      if reg_batch_norm:
         # Batch norm layer 2
         net.add(tf.keras.layers.BatchNormalization())

      # Max pool layer 2: 2x2 window (implicit arguments - padding="valid")
      net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

      # Conv layer 7: 3x3 window, 256 filters
      net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Conv layer 8: 3x3 window, 256 filters
      net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Conv layer 9: 3x3 window, 256 filters
      net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      if reg_batch_norm:
         # Batch norm layer 3
         net.add(tf.keras.layers.BatchNormalization())

      # Max pool layer 3: 2x2 window (implicit arguments - padding="valid")
      net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

      # Conv layer 10: 3x3 window, 256 filters
      net.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Conv layer 11: 3x3 window, 256 filters
      net.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Conv layer 12: 3x3 window, 256 filters
      net.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

      # Max pool layer 3: 2x2 window (implicit arguments - padding="valid")
      net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

      # Flatten layer 1
      net.add(tf.keras.layers.Flatten())

      if reg_wdecay_beta > 0:
         reg_wdecay = tf.keras.regularizers.l2(reg_wdecay_beta)
      else:
         reg_wdecay = None

      # Dense layer 1: 128 neurons
      net.add(tf.keras.layers.Dense(units=128, activation='relu',kernel_regularizer=reg_wdecay))

      # Dense layer 2: 512 neurons
      net.add(tf.keras.layers.Dense(units=512, activation='relu',kernel_regularizer=reg_wdecay))

      if reg_dropout_rate > 0:
         # Dropout layer 1:
         net.add(tf.keras.layers.Dropout(reg_dropout_rate))

      # Dense layer 3: n_classes neurons
      net.add(tf.keras.layers.Dense(units=n_classes,activation='softmax'))

      # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
      # training
      optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
      net.compile(optimizer= optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
      # net.compile()
      # net.summary()
      
      model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
         filepath=checkpoint_save_name,
         save_weights_only=True,
         monitor='val_accuracy',
         mode='max',
         save_best_only=True)


      if data_aug:
         # Crate data generator that randomly manipulates images
         datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            zca_epsilon=1e-06,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest',
            horizontal_flip=True
            # batch_size=batch_size  # Specify the batch size here
         )

         # Configure the data generator for the images in the training sets
         datagen.fit(x_train_normalized)

         # Build the data generator
         train_data_aug = datagen.flow(x_train_normalized, y_train)

         if verbose:
            for x_batch, y_hat_batch in datagen.flow(x_train_sample, y_train_sample, shuffle=False):
               show_methods.show_data_images(images=x_batch.astype('uint8'), labels=y_hat_batch, class_names=class_names,
                                                blocking=False)
               break

         # train_info = net.fit(train_data_aug,
         #                      validation_data=(x_valid_normalized, y_valid),
         #                      epochs=epoch, shuffle=True,
         #                      callbacks=[model_checkpoint_callback],
         #                      class_weight=class_weights_dict)
         train_info = net.fit(train_data_aug,
                              validation_data=(x_valid_normalized, y_valid),
                              epochs=epoch, shuffle=True,
                              callbacks=[model_checkpoint_callback])
      else:
        train_info = net.fit(x_train_normalized, y_train, validation_data=(x_valid_normalized, y_valid),
                              epochs=epoch, shuffle=True, callbacks=[model_checkpoint_callback])
      #   print("done")

   #    # Load the weights of the best model
      print("Loading best save weight from %s..." % checkpoint_save_name)
      net.load_weights(checkpoint_save_name)

      # Save the entire model to file
      print("Saving neural network to %s..." % net_save_name)
      net.save(net_save_name)

      # Save training history to file
      history = train_info.history
      with gzip.open(history_save_name, 'w') as f:
         pickle.dump(history, f)

   # *********************************************************
   # * Training history *
   # *********************************************************

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
   if verbose and confus:
    # Get predicted labels for the test set
    y_pred = net.predict(x_test_normalized)

    # Convert predicted probabilities to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_labels)

   #  # Calculate correct and incorrect predictions per class
   #  correct_predictions = np.diag(conf_matrix)
   #  total_predictions = np.sum(conf_matrix, axis=1)
   #  incorrect_predictions = total_predictions - correct_predictions

   #  # Divide class data into three groups
   #  num_classes = len(class_names)
   #  group_size = (num_classes + 2) // 3  # Divide into three approximately equal groups
   #  class_groups = [class_names[i:i+group_size] for i in range(0, num_classes, group_size)]
   #  correct_groups = [correct_predictions[i:i+group_size] for i in range(0, num_classes, group_size)]
   #  incorrect_groups = [incorrect_predictions[i:i+group_size] for i in range(0, num_classes, group_size)]
   #  class_distribution_groups = [train_class_distribution[i:i+group_size] for i in range(0, num_classes, group_size)]

   #  # Plot each group separately
   #  for group_num, (class_group, correct_group, incorrect_group, class_distribution_group) in enumerate(zip(class_groups, correct_groups, incorrect_groups, class_distribution_groups), start=1):
   #      table_data = [['Flower Name', 'Correct Predictions', 'Incorrect Predictions', 'Class Distribution']]
   #      for class_name, correct_pred, incorrect_pred, class_dist in zip(class_group, correct_group, incorrect_group, class_distribution_group):
   #          table_data.append([class_name, correct_pred, incorrect_pred, f'{class_dist:.2f}'])

   #      plt.figure(figsize=(10, 6))
   #      plt.table(cellText=table_data,
   #                colWidths=[0.1, 0.1, 0.1, 0.1],
   #                cellLoc='center',
   #                loc='center')
   #      plt.axis('off')
   #    #   plt.title(f'Correct and Incorrect Predictions per Class (Group {group_num})')
   #      plt.show()

#    print(len(x_train_normalized))
   if verbose:
      loss_train, accuracy_train = net.evaluate(x_train_normalized, y_train, verbose=0)
      loss_test, accuracy_test = net.evaluate(x_test_normalized, y_test, verbose=0)

      print("Train accuracy (tf): %.2f" % accuracy_train)
      print("Test accuracy  (tf): %.2f" % accuracy_test)

      accuracy_data = {
      'Dataset': ['Train', 'Test'],
      'Accuracy': [accuracy_train, accuracy_test]
      }
   
      fig, ax = plt.subplots()
      ax.axis('tight')
      ax.axis('off')
      ax.table(cellText=[accuracy_data['Accuracy']], colLabels=accuracy_data['Dataset'], loc='center')
      title = 'Train and Test accuracy with weigth decay: %.1e, Drop\nout rate: %.1f,batch norm: %s, Data Augmentation: %s' % (reg_wdecay_beta,reg_dropout_rate, reg_batch_norm, data_aug)
      plt.title(title)

      plt.show()

      # Compute output for 16 test images
      y_test = net.predict(x_test[:16])
      y_test = np.argmax(y_test, axis=1)

      # # Show true labels and predictions for 16 test images
      show_methods.show_data_images(images=x_test[:16],
                                    labels=y_test[:16],predictions=y_test,
                                    class_names=class_names,blocking=True)

   return net

if __name__ == "__main__":

   # Best Model on coarse grained data -- do not change
   ex1(load_from_file=True, verbose=True, opt='adam', epoch = 210, confus= True, fine = False,
             reg_wdecay_beta=0.01, reg_dropout_rate=0.4, reg_batch_norm=True, data_aug=True)
   
   # # # Best Fine grained model
   # ex1(load_from_file=True, verbose=True, opt='adam', epoch = 150, confus = False, fine = True,
   #           reg_wdecay_beta=0.01, reg_dropout_rate=0.4, reg_batch_norm=True, data_aug=True)