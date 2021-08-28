import numpy as np
import os
import xml.etree.cElementTree as ET
import sys
from tqdm import tqdm, trange
import cv2
from PIL import Image
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 batch_size=5,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')} # This dictionary is for internal use.

        self.dataset_size = 0 # As long as we haven't loaded anything yet, the dataset size is zero.
        self.load_images_into_memory = load_images_into_memory
        self.images = None # The only way that this list will not stay `None` is if `load_images_into_memory == True`.
        self.batch_size=batch_size
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.dataset_size/self.batch_size)

    def __getitem__(self, index):
        X,y=self.generate(indices=index)
        return X,y
        
    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes = 'all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False,
                  verbose=True):
            # Set class members.
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        self.include_classes = include_classes

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []


        with open(image_set_filenames) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids
        if verbose: it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filenames)), file=sys.stdout)
        else: it = image_ids
        for image_id in it:
            filename = '{}/{}.jpg'.format(images_dirs, image_id)
            self.filenames.append(filename)
            tree = ET.parse(os.path.join(annotations_dirs, image_id + '.xml'))
            root=tree.getroot()
            boxes = [] # We'll store all boxes for this image here.
            eval_neutr = [] 
            tree.findall('object')
            for obj in root.iter('object'):
                class_name=obj.find('name').text
                if(class_name not in self.include_classes):
                  continue
                class_id=self.classes.index(class_name)
                if(str(class_id) not in self.include_classes):
                    continue
                pose = obj.find('pose').text
                difficult=obj.find('difficult').text
                truncated=obj.find('truncated').text
                bndbox=obj.find('bndbox')
                xmin=bndbox.find('xmin').text
                ymin=bndbox.find('ymin').text
                xmax=bndbox.find('xmax').text
                ymax=bndbox.find('ymax').text
                item_dict={'class_name':class_name,
                           'class_id':class_id, 
                           'pose':pose,
                           'difficult':difficult,
                           'truncated':truncated,
                           'xmin':xmin,
                           'ymin':ymin,
                           'xmax':xmax,
                           'ymax':ymax}
                box=[]
                for item in self.labels_output_format:
                    box.append(item_dict[item])
                boxes.append(box)
            
            self.labels.append(boxes)
            self.eval_neutral.append(eval_neutr)
        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else: it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
        if ret:
            return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral

    
    def generate(self,
                 indices,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove',
                 ):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Can shuffle the samples consistently after each complete pass.

        Optionally takes a list of arbitrary image transformations to apply to the
        samples ad hoc.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list, optional): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple that contains the outputs specified in this set and only those. If an output is not available,
                it will be `None`. The output tuple can contain the following outputs according to the specified keyword strings:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
'''

        batch_X,batch_y=[],[]
    
        # Generate batch.
        batch_indices = indices
        #print(batch_indices)
        batch_X = self.images[batch_indices]
        batch_y = self.labels[batch_indices]
        
            # Apply transformations.
            
        return batch_X,batch_y
    




