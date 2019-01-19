import cv2
import numpy as np
from pandas.io.parsers import read_csv
from cnn_models import basic_model, leNetModel, nVidiaModel
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

def image_brighten(image, base_change = 0.25):
    # change the brightness of the image, using a (random) 
    # multiplication factor
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = np.random.uniform(base_change, 1.0)
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def generator(data, image_path, batch_size=32):
    # Generate the batches for training and validation
    path = image_path 
    num_samples = len(data)
    while 1: 
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]      
            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[1])
                name = image_path+(batch_sample[0].strip()).split('/')[-1]
                image = cv2.imread(name)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)
                
                #Flip the image to augment the data 
                # and avoid bias to clockwise/counterclockwise
                flipped_image = cv2.flip(image, 1)
                images.append(flipped_image)
                angles.append(-angle)
                
                # Augment the data for the classes where abs(angle)>0.5
                if ( abs(angle) > 0.5 ):
                    # Change the brightness of the original image
                    augmented_image = image_brighten(image)
                    images.append(augmented_image)
                    angles.append(angle)
                    
                    # Change the brightness of the flipped image
                    augmented_image = image_brighten(flipped_image)
                    images.append(augmented_image)
                    angles.append(-angle)
            X_data = np.array(images)
            y_data = np.array(angles)    
            yield shuffle(X_data, y_data)

# Read the dataset from the csv file
samples_folder = "data/"
#dataset_samples = read_csv(samples_folder+"driving_log.csv", header=0, usecols=[0,1,2,3,4,5,6]).values;
dataset_samples = read_csv("my_dataset.csv", header=0, usecols=[0,1]).values;

# Split the dataset in train and validation sets
shuffle(dataset_samples)
train_samples, validation_samples = train_test_split(dataset_samples, test_size=0.2)

print(train_samples.shape)
# Create the generators
train_generator = generator(train_samples, samples_folder+"IMG/")
validation_generator = generator(validation_samples, samples_folder+"IMG/")

# Creates and compiles the model
model = nVidiaModel()
model.compile(optimizer= 'adam', loss='mse', metrics=['acc'])

# Name of the model to save
file = 'model.h5'

# Stop training when "val_loss" quantity has stopped improving.
earlystopper = EarlyStopping(patience=5, verbose=1)
#Save the (best) model after every epoch
checkpointer = ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only=True)

# Train the model
print("Trainning")
history_object = model.fit_generator(train_generator, samples_per_epoch = 2*len(train_samples),
                                     validation_data = validation_generator,
                                     nb_val_samples = 2*len(validation_samples), nb_epoch=1, verbose=1)
# Save the model
print("Saving model")
model.save(file)
print("Model Saved")




