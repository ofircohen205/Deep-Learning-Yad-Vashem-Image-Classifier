#############################################################################################################
################################################## IMPORTS ##################################################
#############################################################################################################
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from random import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, math
import getpass
import seaborn as sns; sns.set()

# from trains import Task
# task = Task.init(project_name="DL_CNN_Final_Project", task_name="Test_Model")
# logger = task.get_logger()

##############################################################################################################
################################################## SETTINGS ##################################################
##############################################################################################################
classes = [ 'Animals',
			'Buildings',
			'Carts',
			'Children',
			'Corpses',
			'German Symbols',
			'Gravestones',
			'Railroad cars',
			'Signs',
			'Snow',
			"Uniforms",
			"Vehicles",
			"Views",
			'Weapons',
			'Women',
		]
classes = sorted(classes)

IM_WIDTH, IM_HEIGHT = 150, 150
EPOCHS = 30
BATCH_SIZE = 64*8
FC_SIZE = 2048
NUM_CLASSES = len(classes)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if getpass.getuser() == 'assafsh':
	train_directory = "/mnt/data/Storage/yad-vashem-dataset/data/train"
	validation_directory = "/mnt/data/Storage/yad-vashem-dataset/data/validation"
	test_directory = "/mnt/data/Storage/yad-vashem-dataset/data/test"
else:
	train_directory = os.path.join(BASE_DIR, "data/train")
	validation_directory = os.path.join(BASE_DIR, "data/validation")
	test_directory = os.path.join(BASE_DIR, "data/test")


###############################################################################################################
################################################## FUNCTIONS ##################################################
############################################################################################################### 
def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def generators():
	'''
	This function creates a generator for the dataset - generator for train, generator for validation and generator for test
	For each image in the dataset an augmentation is being executed
	'''
	# Set train and test data generators
	train_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rescale=1./255,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
	)

	test_datagen = ImageDataGenerator(
		rescale=1./255
	)

	# Get images from train directory and insert into generator
	train_generator = train_datagen.flow_from_directory(
		train_directory,
		target_size=(IM_WIDTH, IM_HEIGHT),
		batch_size=BATCH_SIZE,
		shuffle=True,
		class_mode='binary'
	)

	# Get images from validation directory and insert into generator
	validation_generator = test_datagen.flow_from_directory(
		validation_directory,
		target_size=(IM_WIDTH, IM_HEIGHT),
		batch_size=BATCH_SIZE,
		shuffle=True,
		class_mode='binary'
	)

	# Get images from test directory and insert into generator
	test_generator = test_datagen.flow_from_directory(
		test_directory,
		target_size=(IM_WIDTH, IM_HEIGHT),
		batch_size=BATCH_SIZE,
		shuffle=True,
		class_mode='binary'
	)
	return train_generator, validation_generator, test_generator
''' End function '''


def yield_from_generators(train_generator, validation_generator, test_generator):
	
	train_df, validation_df, test_df = [], [], []
	categories_path_train, categories_path_validation, categories_path_test = [], [], []
	
	for category in classes:
		if ' ' in category:
			category = category.replace(" ", "_")
		categories_path_train.append(os.path.join(train_directory, category))
		categories_path_validation.append(os.path.join(validation_directory, category))
		categories_path_test.append(os.path.join(test_directory, category))
	
	for class_num, path in enumerate(categories_path_train):
		dir_path = os.listdir(path)
		for i, child in enumerate(dir_path):
			if i > 79000:
				break
			if i % 100 == 0:
				print("number of train_df: {}". format(len(train_df)))
			img = load_img(os.path.join(path, child), target_size=(IM_HEIGHT, IM_WIDTH, 3))
			x = img_to_array(img)
			train_df.append([x, class_num])
			
	
	for class_num, path in enumerate(categories_path_validation):
		dir_path = os.listdir(path)
		for i, child in enumerate(dir_path):
			if i > 9800:
				break
			if i % 100 == 0:
				print("number of validation_df: {}". format(len(validation_df)))
			img = load_img(os.path.join(path, child), target_size=(IM_HEIGHT, IM_WIDTH, 3))
			x = img_to_array(img)
			validation_df.append([x, class_num])
	
	for class_num, path in enumerate(categories_path_test):
		dir_path = os.listdir(path)
		for i, child in enumerate(dir_path):
			if i > 9800:
				break
			if i % 100 == 0:
				print("number of test_df: {}". format(len(test_df)))
			img = load_img(os.path.join(path, child), target_size=(IM_HEIGHT, IM_WIDTH, 3))
			x = img_to_array(img)
			test_df.append([x, class_num])
	
	shuffle(train_df)

	X_train, X_validation, X_test = [], [], []
	Y_train, Y_validation, Y_test = [], [], []
	
	for image, label in train_df:
		# only divided by 156
		if len(X_train) == 79000:
			break
		X_train.append(image)
		Y_train.append(label)
	
	for image, label in validation_df:
		# only divided by 156
		if len(X_validation) == 9800:
			break
		X_validation.append(image)
		Y_validation.append(label)
	
	for image, label in test_df:
		# only divided by 156
		if len(X_test) == 9800:
			break
		X_test.append(image)
		Y_test.append(label)
	
	X_train = np.array(X_train) / 255.0
	Y_train = np.array(Y_train)
	
	X_validation = np.array(X_validation) / 255.0
	Y_validation = np.array(Y_validation)
	
	X_test = np.array(X_test) / 255.0
	Y_test = np.array(Y_test)
	
	return X_train, X_validation, X_test, Y_train, Y_validation, Y_test
''' End function '''

def generate_class_weights(train_generator):
	'''
	Input:
	Output:
	'''
	labels_dict = {
		'Animals': 1559,
		'Buildings':9052,
		'Carts': 1540,
		'Children': 16525,
		'Corpses': 4606,
		"German Symbols": 2476,
		'Gravestones': 5648,
		'Railroad cars': 1018,
		'Signs': 2038,
		'Snow': 1716,
		"Uniforms": 12356,
		"Vehicles": 3036,
		"Views": 8776,
		'Weapons': 1260,
		'Women': 27642
	}

	class_weights_dict = dict()
	total_samples = sum(labels_dict.values())
	mu = 0.15
	for key in labels_dict.keys():
		score = math.log(mu * total_samples / float(labels_dict[key]))
		class_weights_dict[classes.index(key)] = score if score > 1.0 else 1.0
	
	print(class_weights_dict)
	return class_weights_dict
''' End function '''


def create_classifier(base_model):
	'''
	Creates new classifiers based on ResNet50
	'''
	# Add global average pooling and 2 FC for fine tuning
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(FC_SIZE, activation='relu')(x)
	x = Dense(FC_SIZE//2, activation='relu')(x)
	predictions = Dense(NUM_CLASSES, activation='softmax')(x)

	# Create the model
	model = Model(base_model.input, predictions)
	return model
''' End function '''


def fit_predict(X_train, X_validation, X_test, Y_train, Y_validation, Y_test, train_generator, validation_generator, test_generator, classifier, class_weight_dict, number):
	'''
	Input:
	Output:
	'''    
	history = classifier.fit(
		X_train,
		Y_train,
		steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
		epochs=EPOCHS,
		validation_data=(X_validation, Y_validation),
		validation_steps=X_validation.shape[0] // BATCH_SIZE,
		shuffle=True,
		callbacks=[tf.keras.callbacks.CSVLogger('training_{}.log'.format(number))],
		class_weight=class_weight_dict,
		use_multiprocessing=True,
		workers=8,
	)
	
	classifier.save_weights('train_without_base_model.h5')
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('./plots/accuracy_plot_{}.png'.format(number))
	plt.clf()
	
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title("model loss")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('./plots/loss_plot_{}.png'.format(number))
	plt.clf()
	print("====================================================")

	history_evaulate = classifier.evaluate(X_validation, Y_validation)
	print("model evaulation on test:")
	print(history_evaulate)
	print("====================================================")
	Y_pred = classifier.predict(X_test)
	y_pred = np.argmax(Y_pred, axis=1)
	
	print("====================================================")    
	print("Confusion matrix:")
	conf = confusion_matrix(Y_test, y_pred)
	print(conf)
	plt.figure(figsize=(20,20))
	ax = plt.axes()
	sns.heatmap(conf, ax=ax, xticklabels=classes, yticklabels=classes, linewidths=0.5, annot=True, fmt='d')
	ax.set_title('Confunsion Matrix')
	b,t = plt.ylim()
	b += 0.5
	t -= 0.5
	plt.ylim(b,t)
	plt.savefig('./plots/confusion_matrix{}.png'.format(number))
    plt.clf()
	print("====================================================")    
	print("Classification report:")
	class_report = classification_report(Y_test, y_pred, target_names=classes)
	print(class_report)
	with open("classification_report{}.log".format(number), 'w') as f:
		f.write(class_report)
''' End function '''


##########################################################################################################
################################################## MAIN ##################################################
##########################################################################################################
def main():

	strategy = tf.distribute.MirroredStrategy()
	with strategy.scope():
		train_generator, validation_generator, test_generator = generators()
		class_weight_dict = generate_class_weights(train_generator)
		X_train, X_validation, X_test, Y_train, Y_validation, Y_test = yield_from_generators(train_generator, validation_generator, test_generator)
		
		# Set ResNet to be base model
		base_model = ResNet152V2(weights="imagenet", include_top=False)
		classifier = create_classifier(base_model)
		
		# Freeze all base model layers
		for layer in base_model.layers:
			layer.trainable = False

		classifier.compile(optimizer=Adam(), loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=['accuracy'])
		classifier.summary()
		
		print("Transfer learning")
		fit_predict(X_train, X_validation, X_test, Y_train, Y_validation, Y_test, train_generator, validation_generator, test_generator, classifier, class_weight_dict, 0)
		
		# Unfreeze all base model layers
		for layer in base_model.layers:
			layer.trainable = True
		
		classifier.compile(optimizer=Adam(), loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=['accuracy'])
		classifier.summary()
		
		print("Fine Tuning")
		fit_predict(X_train, X_validation, X_test, Y_train, Y_validation, Y_test, train_generator, validation_generator, test_generator, classifier, class_weight_dict, 1)


if __name__ == "__main__":
	main()
