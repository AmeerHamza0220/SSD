import tensorflow as tf
from tensorflow.keras import layers

def conv2d(x,filters,filter_size,strides=1,padding='same',name=None,activation='relu'):
  x=layers.Conv2D(filters,filter_size,strides=strides,
      padding=padding,
    name=name)(x)
  if activation is not None:
    ac_name = None if name is None else name + '_ac'
    x = layers.Activation(activation, name=ac_name)(x)
  return x


def InceptionResnet(input_tensor=None,
                    classes=5,
                    input_shape=None,
                    classifier_activation="softmax"):
  if input_tensor is None:
    img_input = tf.keras.layers.Input(shape=input_shape)
  else:
    img_input = input_tensor

  #stem
  x=conv2d(img_input,32,3,strides=2,padding="valid")
  x=conv2d(x,32,3)
  x=conv2d(x,64,3)
  branch_0 = layers.MaxPooling2D(3, strides=2)(x)
  branch_1=conv2d(x,96,3,strides=2,padding="valid")
  channel_axis = 3
  branches=[branch_0,branch_1]
  x = layers.Concatenate(axis=channel_axis, name='stem_1')(branches)
  #branch 2
  branch_0=conv2d(x,64,1)
  branch_0=conv2d(branch_0,64,[7,1])
  branch_0=conv2d(branch_0,64,[1,7])
  branch_0=conv2d(branch_0,96,3)
  branch_1=conv2d(x,64,1)
  branch_1=conv2d(branch_1,96,3)
  branches=[branch_0,branch_1]
  x = layers.Concatenate(axis=channel_axis, name='stem_2')(branches)
  #branch_3
  branch_0=conv2d(x,192,3,strides=1,padding='valid')
  print(branch_0.shape)
  branch_1 = layers.MaxPooling2D(3,strides=1)(x)
  branches=[branch_0,branch_1]
  x = layers.Concatenate(axis=channel_axis, name='stem_3')(branches)

  #inception A block
  branch_1=conv2d(x,96,1)
  branch_2=conv2d(x,64,1)
  branch_2=conv2d(branch_2,96,3)

  branch_3=conv2d(x,64,1)
  branch_3=conv2d(branch_3,96,3)
  branch_3=conv2d(branch_3,96,3)
  branch_0=layers.AveragePooling2D(3,strides=1,padding='same')(x)
  branch_0=conv2d(branch_0,96,1)

  branches=[branch_0,branch_1,branch_2,branch_3]
  x = layers.Concatenate(axis=channel_axis, name='block_A')(branches)
  
#inception Reduction_A block
  
  branch_1=conv2d(x,384,3,strides=2,padding='valid')
  branch_2=conv2d(x,256,1)
  branch_2=conv2d(branch_2,256,3)

  branch_2=conv2d(branch_2,384,3,strides=2,padding='valid')

  
  branch_0=layers.MaxPooling2D(3,strides=2,padding='valid')(x)

  branches=[branch_0,branch_1,branch_2]
  x = layers.Concatenate(axis=channel_axis, name='reduction_block_A')(branches)
  
#inception B block
  branch_1=conv2d(x,384,1)
  branch_2=conv2d(x,192,1)
  branch_2=conv2d(branch_2,224,[1,7])
  branch_2=conv2d(branch_2,256,[1,7])

  branch_3=conv2d(x,192,1)
  branch_3=conv2d(branch_3,192,[1,7])
  branch_3=conv2d(branch_3,224,[7,1])
  branch_3=conv2d(branch_3,224,[1,7])
  branch_3=conv2d(branch_3,256,[7,1])

  branch_0=layers.AveragePooling2D(3,strides=1,padding='same')(x)
  branch_0=conv2d(branch_0,128,1)

  branches=[branch_0,branch_1,branch_2,branch_3]
  x = layers.Concatenate(axis=channel_axis, name='block_B')(branches)
  
#inception Reduction_B block
  branch_1=conv2d(x,192,3)
  branch_1=conv2d(x,192,3,strides=2,padding='valid')

  branch_2=conv2d(x,256,1)
  branch_2=conv2d(branch_2,256,[1,7])
  branch_2=conv2d(branch_2,320,[7,1])

  branch_2=conv2d(branch_2,320,3,strides=2,padding='valid')

  
  branch_0=layers.MaxPooling2D(3,strides=2,padding='valid')(x)

  branches=[branch_0,branch_1,branch_2]
  x = layers.Concatenate(axis=channel_axis, name='reduction_block_B')(branches)
  
#inception C block
  branch_1=conv2d(x,256,1)
  branch_2=conv2d(x,384,1)
  branch_2_1=conv2d(branch_2,256,[1,3])
  branch_2_2=conv2d(branch_2,256,[3,1])
  branch_2_3=[branch_2_1,branch_2_2]
  branch_2 = layers.Concatenate(axis=channel_axis, name='block_C_1')(branch_2_3)


  branch_3=conv2d(x,384,1)
  branch_3=conv2d(branch_3,448,[1,3])
  branch_3=conv2d(branch_3,512,[3,1])
  branch_3_1=conv2d(branch_3,256,[3,1])
  branch_3_2=conv2d(branch_3,256,[1,3])
  branch_3=[branch_3_1,branch_3_2]
  branch_3 = layers.Concatenate(axis=channel_axis, name='block_C_2')(branch_3)

  branch_0=layers.AveragePooling2D(3,strides=1,padding='same')(x)
  branch_0=conv2d(branch_0,256,1)

  branches=[branch_0,branch_1,branch_2,branch_3]
  x = layers.Concatenate(axis=channel_axis, name='block_C')(branches)
  

  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = layers.Dropout(0.2)(x)


  return x