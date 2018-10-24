from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import GaussianNoise
from tensorflow.python.keras import regularizers
import os.path

MODEL_FILE = "cat_5.hd5"

# Create a model if none exists. Freezes all training except in
# newly attached output layers. We can specify the number of nodes
# in the hidden penultimate layer, and the number of output
# categories.output


def create_model(num_hidden, num_classes):
	base_model = InceptionV3(include_top = False, weights = 'imagenet')
	x = base_model.output
	#x = GaussianNoise(0.1)(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(num_hidden, activation = 'relu', kernel_regularizer=regularizers.l2(0), activity_regularizer=regularizers.l1(0.0003))(x)
	predictions = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l2(0), activity_regularizer=regularizers.l1(0.0003))(x)

	for layer in base_model.layers:
		layer.trainable = False

	model = Model(inputs = base_model.input, outputs = predictions)

	return model

def train(model_file, train_path, validation_path, num_hidden=200, num_classes=4, steps=32, num_epochs=20, save_period=1):

	if os.path.exists(model_file):
		print ("\n*** Existing model found at {}. Loading.***\n\n".format(model_file) )
		model = create_model(num_hidden, num_classes)

	else:
		print ("\n*** Creating new model ***\n\n")
		model = create_model(num_hidden, num_classes)

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
	stopmodel = EarlyStopping(min_delta = 0.01, patience = 5)
	#Create a checkpoint
	checkpoint = ModelCheckpoint(model_file, period = save_period)

	train_datagen = ImageDataGenerator(\
		rescale=1./255,\
		shear_range=0.2,\
		zoom_range=0.2,\
		horizontal_flip = True)
	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(\
		train_path,\
		target_size = (249,249),
		batch_size=32,
		class_mode = "categorical")

	validation_generator = test_datagen.flow_from_directory(\
		validation_path,\
		target_size = (249,249),
		batch_size = 32,
		class_mode = "categorical")

	model.fit_generator(\
		train_generator,\
		steps_per_epoch = steps,\
		epochs = num_epochs,\
		callbacks = [checkpoint, stopmodel],\
		validation_data = validation_generator,\
		validation_steps = 50)

	#Train last two layers
	for layer in model.layers[:249]:
		layer.trainable = False

	for  layer in model.layers[249:]:
		layer.trainable = True

	model.compile(optimizer=SGD(lr=0.0005, momentum=0.9), loss='categorical_crossentropy')

# Load an existing model file, then sets only the last
# 3 layers (which we added) to trainable

def load_existing(model_file):
	# Load the model
	model = load_model(model_file)

	# Set only last 3 layers as trainable
	numlayers = len(model.layers)

	for layer in model.layers[:numlayers - 3]:
		layer.trainable = False

	for layer in model.layers[numlayers - 3:]:
		layer.trainable = True

	return model


def main():
	train(MODEL_FILE, train_path = "cat_train", validation_path = "cat_test", steps = 20, num_epochs = 10)

if __name__ == "__main__":
	main()
