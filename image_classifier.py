# Load Imagenet Base Model
from tensorflow import keras

base_model = keras.applications.VGG16(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False)

base_model.trainable = False

# Add Layers to Model -------------------------------------------------
# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(1, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

model.summary()

# Compile Model -------------------------------------------------
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Augment the data -------------------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen_train = ImageDataGenerator(FIXME)
datagen_train = ImageDataGenerator(
    samplewise_center=True,  # set each sample mean to 0
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
)

# datagen_valid = ImageDataGenerator(FIXME)
datagen_valid = ImageDataGenerator(samplewise_center=True)


# Load Dataset -------------------------------------------------

# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    "data/fruits/train/",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
)

# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    "data/fruits/valid/",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
)

# Train the Model -------------------------------------------------
model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples//train_it.batch_size,
          validation_steps=valid_it.samples//valid_it.batch_size,
          epochs=20)
