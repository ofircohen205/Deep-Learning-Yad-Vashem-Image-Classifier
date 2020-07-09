#############################################################################################################
################################################## IMPORTS ##################################################
#############################################################################################################
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from random import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import getpass


##############################################################################################################
################################################## SETTINGS ##################################################
##############################################################################################################
classes = [ 'Women', 'Children', 'Animals', 'Uniforms', 'Buildings', 'Street scene',
            'Vehicles', 'Signs', 'Weapons', 'Railroad cars', 'Nazi symbols', 'Gravestones',
            'Barbed wire fences', 'Corpses', 'German soldiers', 'Armband', 'Snow', 'Carts',
        ]
classes = sorted(classes)

IM_WIDTH, IM_HEIGHT = 224, 224
EPOCHS = 50
BS = 32
FC_SIZE = 2048
NUM_CLASSES = len(classes)
LAYERS_TO_FREEZE = 249
SEED=42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if getpass.getuser() == 'assafsh':
    train_directory = "/mnt/data/Storage/DeepLearningFinalProject/data/train"
    validation_directory = "/mnt/data/Storage/DeepLearningFinalProject/data/validation"
    test_directory = "/mnt/data/Storage/DeepLearningFinalProject/data/test"
else:
    train_directory = os.path.join(BASE_DIR, "data/train")
    validation_directory = os.path.join(BASE_DIR, "data/validation")
    test_directory = os.path.join(BASE_DIR, "data/test")


###############################################################################################################
################################################## FUNCTIONS ##################################################
############################################################################################################### 
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy

def generators():
    '''
    This function creates a generator for the dataset - generator for train, generator for validation and generator for test
    For each image in the dataset an augmentation is being executed
    '''
    # Set train and test data generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Get images from train directory and insert into generator
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BS,
        shuffle=True,
        seed=SEED
    )

    # Get images from validation directory and insert into generator
    validation_generator = test_datagen.flow_from_directory(
        validation_directory,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BS,
        shuffle=True,
        seed=SEED
    )

    # Get images from test directory and insert into generator
    test_generator = test_datagen.flow_from_directory(
        test_directory,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BS,
        shuffle=True,
        seed=SEED
    )
    return train_generator, validation_generator, test_generator
''' End function '''


def generate_class_weights(train_generator):
    '''
    Input:
    Output:
    '''
    X, Y = train_generator.next()
    # Instantiate the label encoder
    le = LabelEncoder()

    # Fit the label encoder to our label series
    le.fit(list(classes))

    # Create integer based labels Series
    y_integers = le.transform(list(classes))

    #print y_integers
    # Create dict of labels : integer representation
    labels_and_integers = dict(zip(classes, y_integers))

    print(labels_and_integers)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)

    class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))
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


def fit_predict(train_generator, validation_generator, test_generator, classifier, class_weight_dict):
    '''
    Input:
    Output:
    '''
    history_without_base_model = classifier.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        class_weight=class_weight_dict
    )
    
    classifier.save_weights('train_without_base_model.h5')
    print("====================================================")

    history_without_base_model_return_value = classifier.evaluate_generator(test_generator)
    print("model evaulation on test:")
    print(history_without_base_model_return_value)
    print("====================================================")

    for layer in classifier.layers[:LAYERS_TO_FREEZE]:
        layer.trainable = False
        
    for layer in classifier.layers[LAYERS_TO_FREEZE:]:
        layer.trainable = True

    # Set optimizer Adam and loss function to be CategoricalCrossentropy
    classifier.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

    history = classifier.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        class_weight=class_weight_dict
    )
    classifier.save_weights('train.h5')
    print("====================================================")

    history_return_value = classifier.evaluate_generator(test_generator)
    print("model evaulation on test:")
    print(history_return_value)
    print("====================================================")
''' End function '''


def fit_predict_overfitting(classifier, number):
    '''
    Input: classifier
    Output: train on 80 images per class, validate on 10 and test on 10.
    '''
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
            if i == 80:
                break
            img = load_img(os.path.join(path, child), target_size=(IM_HEIGHT, IM_WIDTH, 3))
            x = img_to_array(img)
            train_df.append([x, class_num])
            
    
    for class_num, path in enumerate(categories_path_validation):
        dir_path = os.listdir(path)
        for i, child in enumerate(dir_path):
            if i == 10:
                break
            img = load_img(os.path.join(path, child), target_size=(IM_HEIGHT, IM_WIDTH, 3))
            x = img_to_array(img)
            validation_df.append([x, class_num])
    
    for class_num, path in enumerate(categories_path_test):
        dir_path = os.listdir(path)
        for i, child in enumerate(dir_path):
            if i == 10:
                break
            img = load_img(os.path.join(path, child), target_size=(IM_HEIGHT, IM_WIDTH, 3))
            x = img_to_array(img)
            test_df.append([x, class_num])
    
    shuffle(train_df)
    shuffle(validation_df)
    shuffle(test_df)

    X_train, X_validation, X_test = [], [], []
    Y_train, Y_validation, Y_test = [], [], []
    
    for image, label in train_df:
        X_train.append(image)
        Y_train.append(label)
    
    for image, label in validation_df:
        X_validation.append(image)
        Y_validation.append(label)
    
    for image, label in test_df:
        X_test.append(image)
        Y_test.append(label)
    
    X_train = np.array(X_train) / 255.0
    Y_train = np.array(Y_train)
    
    X_validation = np.array(X_validation) / 255.0
    Y_validation = np.array(Y_validation)
    
    X_test = np.array(X_test) / 255.0
    Y_test = np.array(Y_test)
    
    history = classifier.fit(
        X_train,
        Y_train,
        steps_per_epoch=X_train.shape[0] // BS,
        epochs=EPOCHS,
        validation_data=(X_validation, Y_validation),
        validation_steps=X_validation.shape[0] // BS,
        shuffle=True
    )
    classifier.save_weights('train_overfitting.h5')
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
    
    history_without_base_model_return_value = classifier.evaluate(X_validation, Y_validation)
    print("model evaulation on test:")
    print(history_without_base_model_return_value)
''' End function '''


##########################################################################################################
################################################## MAIN ##################################################
##########################################################################################################
def main():
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
    fit_predict_overfitting(classifier, 0)
    
    for layer in base_model.layers:
        layer.trainable = True
    
    classifier.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    classifier.summary()
    
    print("Fine Tuning")
    fit_predict_overfitting(classifier, 1)
    
    # # Freeze all base model layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    # classifier.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    # classifier.summary()
    
    # print("Transfer learning")
    # fit_predict(train_generator, validation_generator, test_generator, classifier, class_weight_dict)
    
    # for layer in classifier.layers:
    #     layer.trainable = True
    
    # classifier.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    # classifier.summary()
    
    # fit_predict(train_generator, validation_generator, test_generator, classifier, class_weight_dict)


if __name__ == "__main__":
    main()