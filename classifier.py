#############################################################################################################
################################################## IMPORTS ##################################################
#############################################################################################################
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet152V2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy, Poisson
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

IM_WIDTH, IM_HEIGHT = 224, 224
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
    total_samples = train_generator.n
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


def fit_predict(train_generator, validation_generator, test_generator, classifier, class_weight_dict, number):
    '''
    Input:
    Output:
    '''
    history = classifier.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
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

    history_evaulate = classifier.evaluate(validation_generator)
    print("model evaulation on test:")
    print(history_evaulate)
    print("====================================================")
    Y_pred = classifier.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print("====================================================")    
    print("Confusion matrix:")
    print(confusion_matrix(test_generator.classes, y_pred))
    with open("confusion_matrix_{}".format(number), 'w') as f:
        f.write(confusion_matrix(test_generator.classes, y_pred))
    print("====================================================")    
    print("Classification report:")
    print(classification_report(test_generator.classes, y_pred, target_names=classes))
    with open("classification_report{}".format(number), 'w') as f:
        f.write(classification_report(test_generator.classes, y_pred, target_names=classes))
''' End function '''


##########################################################################################################
################################################## MAIN ##################################################
##########################################################################################################
def main():
    tf.debugging.set_log_device_placement(True)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        train_generator, validation_generator, test_generator = generators()
        class_weight_dict = generate_class_weights(train_generator)
        
        # Set ResNet to be base model
        base_model = ResNet50V2(weights="imagenet", include_top=False)
        classifier = create_classifier(base_model)
        
        # Freeze all base model layers
        for layer in base_model.layers:
            layer.trainable = False

        classifier.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        classifier.summary()
        
        print("Transfer learning")
        fit_predict(train_generator, validation_generator, test_generator, classifier, class_weight_dict, 0)
        
        # Unfreeze all base model layers
        for layer in base_model.layers:
            layer.trainable = True
        
        classifier.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        classifier.summary()
        
        fit_predict(train_generator, validation_generator, test_generator, classifier, class_weight_dict, 1)


if __name__ == "__main__":
    main()