__author__ = "Lech Szymanski"
__organization__ = "COSC420, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle, gzip
import progressbar
import os

"""
The original Oxford flowers102 dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) consists of colour images of 102 
different types of flowers with 40 and 258 example images per each class.

This dataaset provides a train, validation, and test split of the data in one of three possible formats: numpy (default), tfds, or pandas.

The numpy format provides two options:
- fine labelled set: provides original dataset with 102 classes, 6149 training images, 1020 validation images, and 1020 test images.
- coarse labelled set: which is a subset of the fine labelled set with selected flowers grouped into 10 classes.
and it formats all images to specified square size (default is 96x96 pixel, maximum possible 500x500).

The tdfs and pandas datasets return the original Oxford flowers102 dataset (fine labelled one) in the tfds and pandas data frame format, respectively (https://www.tensorflow.org/datasets/catalog/oxford_flowers102)

In the numpy array format, the data structure format for train or validation or test is as follows:
- data['images'] - an N x imsize x imsize x 3 numpy array of N images of imsize x imsize pixels in RGB format;
- data['labels'] - an N x 1 numpy array of N labels (0-101 for fine labelled, 0-9 for coarase labelled) corresponding to the type (for fine labeleld) or group (for corase labelled) of flower in the image;
- data['file_names'] - a list of N byte strings, each is the name of the file (in the original dataset) from which the corresponding image was loaded.

To use this script in another, just drop this file in the same folder and then, you can invoke it from the other
script like so:

from load_smallnorb import load_smallnorb

train_data, validation_data, test_data, class_names = load_oxford_flowers102(imsize=96, fine=False)

See example below for use of the load_oxford_flowers102 method.

"""



"""
Labels for the Oxford flowers102 dataset (fine labelled set)
"""
flowers102_class_names = ['pink primrose','hard-leaved pocket orchid', 'canterbury bells','sweet pea','english marigold',
                          'tiger lily','moon orchid','bird of paradise','monkshood','globe thistle','snapdragon',"colt's foot",
                          'king protea','spear thistle','yellow iris','globe-flower','purple coneflower','peruvian lily',
                          'balloon flower','giant white arum lily','fire lily','pincushion flower','fritillary','red ginger',
                          'grape hyacinth','corn poppy',"prince of wales feathers",'stemless gentian','artichoke','sweet william',
                          'carnation','garden phlox','love in the mist','mexican aster','alpine sea holly','ruby-lipped cattleya',
                          'cape flower','great masterwort','siam tulip','lenten rose','barbeton daisy','daffodil','sword lily',
                          'poinsettia','bolero deep blue','wallflower','marigold','buttercup','oxeye daisy','common dandelion',
                          'petunia','wild pansy','primula','sunflower','pelargonium','bishop of llandaff','gaura','geranium',
                          'orange dahlia','pink-yellow dahlia','cautleya spicata','japanese anemone','black-eyed susan',
                          'silverbush','californian poppy','osteospermum','spring crocus','bearded iris','windflower',
                          'tree poppy','gazania','azalea','water lily','rose','thorn apple','morning glory','passion flower',
                          'lotus','toad lily','anthurium','frangipani','clematis','hibiscus','columbine','desert-rose',
                          'tree mallow','magnolia','cyclamen','watercress','canna lily','hippeastrum','bee balm','ball moss',
                          'foxglove','bougainvillea','camellia','mallow','mexican petunia','bromelia','blanket flower',
                          'trumpet creeper','blackberry lily'
]

"""
Grouping for coarse labelled set
"""
flowers102_group_names = {
    "Orchids": [ "hard-leaved pocket orchid","moon orchid","ruby-lipped cattleya"],
    "Bell-shaped Flowers": [ "canterbury bells","bluebell","campanula"],
    "Lilies": ["tiger lily","fire lily","giant white arum lily","toad lily","canna lily"],
    "Tubular Flowers": ["snapdragon","trumpet creeper","foxglove","bee balm"],
    "Composite Flowers": ["daisy","sunflower","marigold","black-eyed susan","english marigold","purple coneflower",
                          "barbeton daisy","oxeye daisy"],
    "Iris-like Flowers": ["yellow iris","bearded iris","siberian iris","blackberry lily",  "sword lily"],
    "Dahlia Varieties": ["orange dahlia","pink-yellow dahlia","ball dahlia"],
    "Poppies": ["corn poppy","californian poppy","iceland poppy"],
    "Water Flowers": ["water lily","lotus","water hyacinth"],
    "Carnations": ["sweet william","carnation"]
}

def load_oxford_flowers102(format='numpy', imsize=96, fine=False):
   """
   Loads the flowers102 dataset
   Arguments:
       format: a string ('numpy' (default), 'tfds', or 'pandas')
       imsize: an integer (default is 96) specifying the size of the images to be returned
       fine: a boolean (default is False) specifying whether to return the fine labelled set (True) or the coarse labelled set (False)

    Returns:

      when format=='numpy':
         Tuples (train_data, validation_data, test_data, class_names) where train_data, validation_data, and test_data are dictionaries
         that contain 'images' and 'labels', and the class_names is a list of strings containing the class names.

      when format=='tfds':
         A tuple (oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test) containing the original (fine labelled) train, validation and test dataset in tfds format;

      when format=='pandas':
         A tuple (oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test) containing the original (fine labelled) train, validation and test dataset in pandas data frame format.
   """

   numpy_save = 'data/oxford_flowers102_%dx%d.data' % (imsize,imsize)

   if fine:
       k_keep = 'fine'
       k_remove = 'coarse'
   else:
       k_keep = 'coarse'
       k_remove = 'fine'

   # If request format is 'numpy' and 'smallnorb.data' file exists,
   # load it (speeds up the loading).
   if format=='numpy' and os.path.isfile(numpy_save):
      with gzip.open(numpy_save) as f:
         oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test, class_names = pickle.load(f)
        

      oxford_flowers102_train = labelling_select(oxford_flowers102_train,k_keep,k_remove)
      oxford_flowers102_validation = labelling_select(oxford_flowers102_validation,k_keep,k_remove)
      oxford_flowers102_test = labelling_select(oxford_flowers102_test,k_keep,k_remove)

      return oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test, class_names[k_keep]

   group_names = list(flowers102_group_names.keys())
   fine_to_group = -np.ones(len(flowers102_class_names)).astype('int8')

   for i in range(len(flowers102_class_names)):
      for j, group_name in enumerate(group_names):
         if flowers102_class_names[i] in flowers102_group_names[group_name]:
            fine_to_group[i] = j

   # Load the smallnorb dataset in tfds format
   oxford_flowers102  = tfds.load(name="oxford_flowers102", split=None)
   oxford_flowers102_train = oxford_flowers102['test']
   oxford_flowers102_validation = oxford_flowers102['validation']
   oxford_flowers102_test = oxford_flowers102['train']

   # If tfds format requested, return the data
   if format == 'tfds':
      return oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test

   # Convert data to pandas data frame
   oxford_flowers102_train = tfds.as_dataframe(oxford_flowers102_train)
   oxford_flowers102_validation = tfds.as_dataframe(oxford_flowers102_validation)
   oxford_flowers102_test = tfds.as_dataframe(oxford_flowers102_test)

   # If pandas format requested, return the data
   if format == 'pandas':
      return oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test

   # Conert pandas frame to numpy
   oxford_flowers102_train = oxford_flowers102_train.to_numpy()
   oxford_flowers102_validation = oxford_flowers102_validation.to_numpy()
   oxford_flowers102_test = oxford_flowers102_test.to_numpy()

   # Fetch the data out of the frames
   print("Converting training images to %dx%d numpy format..." % (imsize,imsize))
   oxford_flowers102_train = pandas_numpy_to_data_dict(oxford_flowers102_train,6149,fine_to_group,imsize)
   print("Converting validation images to %dx%d numpy format..." % (imsize,imsize))
   oxford_flowers102_validation = pandas_numpy_to_data_dict(oxford_flowers102_validation,1020,fine_to_group,imsize)
   print("Converting test images to %dx%d numpy format..." % (imsize,imsize))
   oxford_flowers102_test = pandas_numpy_to_data_dict(oxford_flowers102_test,1020,fine_to_group,imsize)

   class_names = dict()
   class_names['coarse'] = group_names
   class_names['fine'] = flowers102_class_names

   # Save numpy data to 'smallnorb.data' for fast loading
   print("Caching data to %s for faster loading..." % (numpy_save))
   with gzip.open(numpy_save, 'w') as f:
      pickle.dump((oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test,class_names), f)

   oxford_flowers102_train = labelling_select(oxford_flowers102_train,k_keep,k_remove)
   oxford_flowers102_validation = labelling_select(oxford_flowers102_validation,k_keep,k_remove)
   oxford_flowers102_test = labelling_select(oxford_flowers102_test,k_keep,k_remove)

   return oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test, class_names[k_keep]

def labelling_select(data,k_keep,k_remove):
    """
    Helper function to remove one set of labels from the data, depending if
    we want fine grained or coarse grained labels.
    """

    del data['labels_' + k_remove]
    data['labels'] = data.pop('labels_' + k_keep)

    if k_keep=='coarse':
        I = np.where(data['labels'] >= 0)[0]
        data['images'] = data['images'][I]
        data['labels'] = data['labels'][I].astype('uint8')
        data['file_names'] = [data['file_names'][i] for i in I]

    return data

def pandas_numpy_to_data_dict(X,N,fine_to_group,imsize=96):
    """
    Helper function that converts pandas data frame to numpy data dictionary
    """

    data = dict()
    data['file_names'] = []
    data['images'] = np.zeros((N,imsize,imsize,3)).astype('uint8')
    data['labels_coarse'] = np.zeros((N)).astype('int8')
    data['labels_fine'] = np.zeros((N)).astype('uint8')

    for n in progressbar.progressbar(range(N)):
        data['file_names'].append(X[n,0])
        im = X[n,1]
        H,W,_ = np.shape(im)
        I = np.min([H,W])

        if I < H:
           im = im[int((H-I)/2):int((H-I)/2)+I,:,:]
        elif I < W:
           im = im[:,int((W-I)/2):int((W-I)/2)+I,:]

        # Reshape im to imsize
        data['images'][n] = tf.image.resize(im, (imsize,imsize), method='nearest')

        data['labels_fine'][n] = X[n,2]
        data['labels_coarse'][n] = fine_to_group[X[n,2]]

    return data

def write_to_folder(data, folder_name, class_names=None):
    """
    Helper function that writes images from a data to folder
    """


    from PIL import Image  

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name, exist_ok=True)


    filenames_to_label = []
    print("Writing images to %s..." % folder_name)
    for i in progressbar.progressbar(range(len(data['images']))):
        im_file = data['file_names'][i].decode()
        im_path = os.path.join(folder_name, im_file)
        im_label = data['labels'][i]
        filenames_to_label.append((im_file,im_label))
        im = Image.fromarray(data['images'][i])
        im.save(im_path)      

    with open(os.path.join(folder_name, 'labels.txt'), 'w') as f:
        f.write("Filename, ID\n")
        for file_name,id in filenames_to_label:
            f.write("%s, %d\n" % (file_name,id))

    if class_names is not None:
        with open(os.path.join(folder_name, 'class_names.txt'), 'w') as f:
            f.write("ID, class_name\n")
            for i,class_name in enumerate(class_names):
                f.write("%d, %s\n" % (i,class_name))


if __name__ == "__main__":
    import show_methods


    # Load the smallNORB dataset in numpy format
    train_data, validation_data, test_data, class_names = load_oxford_flowers102(imsize=64, fine=False)

    # You can write the images to folder to inspect them
    write_to_folder(train_data, 'oxford_flowers102_train', class_names)

    # Show 16 train images with the corresponding labels
    show_methods.show_data_images(images=train_data['images'][:16],labels=train_data['labels'][:16], class_names=class_names,blocking=True)

