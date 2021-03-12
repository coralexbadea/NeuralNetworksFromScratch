from layers import Dense, ReLU, Softmax_CELoss, SGD
from preprocess import show_image, labels_to_logits, augmentImages
from imutils import paths
import numpy as np
import argparse
import os
import cv2
import PIL.Image as pil
import sys
import gc
import time 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to the output plot")
args = vars(ap.parse_args())

#initialize the list of paths of the images in the dataset 
imagePaths = list(paths.list_images(args["dataset"]))

data = []
labels = []
labels_uniques = []
imgDim = (224,224)
n=0

for imagePath in imagePaths:
	
	# extract the lable from the path name 
	label = imagePath.split(os.path.sep)[-2]
	# load the input image (224x224) and preprocess it
	print(label)
	image = pil.open(imagePath).convert('L')
	
	image = image.resize(imgDim, pil.ANTIALIAS)
	
	#resized = cv2.resize(np.asarray(image), imgDim, interpolation = cv2.INTER_AREA)
	data.append(np.asarray(image))
	image.close()
	#print(n)
	labels.append(label)
	

data = np.array(data)
for i in range(10):
	print(data[i].shape)

sys.exit(0)
data = data/255



labels_uniques = np.unique(labels)
labels_logits = labels_to_logits(labels)

print(data.shape)
def split_train_test(data, labels, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]
print(labels_logits.shape)

train_data, train_labels, test_data, test_labels = split_train_test(data, labels_logits, 0.25);
#(train_data, train_labels) = augmentImages(train_data, train_labels, "verticalflip", "horizontalflip", "rotation")
n = gc.collect()

#for i in range(4):
#	show_image(train_data[i], str(train_labels[i]))
#sys.exit(0)
#Dense

size_trainset = train_data.shape[0]
size_testset = test_data.shape[0]

train_data = train_data.reshape(size_trainset,-1)
test_data = test_data.reshape(size_testset,-1)

print(train_labels.shape)
print(train_labels.dtype)

print(train_labels.dtype)



inputs = train_data
dense1 = Dense(inputs.shape[1],64)
activation1 = ReLU()
dense2 = Dense(64,128)
activation2 = ReLU()
dense3 = Dense(128,64)
activation3 = ReLU()
dense4 = Dense(64,len(labels_uniques))
activation_and_loss = Softmax_CELoss()
optimizer = SGD(learning_rate = 0.1)



def iterate(inputs, labels):
	dense1.forward(inputs)
	activation1.forward(dense1.outputs)
	dense2.forward(activation1.outputs)
	activation2.forward(dense2.outputs)
	dense3.forward(activation2.outputs)
	activation3.forward(dense3.outputs)
	dense4.forward(activation3.outputs)
	activation_and_loss.forward(dense4.outputs, labels)


	activation_and_loss.backward(labels)
	dense4.backward(activation_and_loss.dinputs)
	activation3.backward(dense4.dinputs)
	dense3.backward(activation3.dinputs)
	activation2.backward(dense3.dinputs)
	dense2.backward(activation2.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.update_params(dense3)
	optimizer.update_params(dense4)
	
def accuracy(inputs, labels):
	dense1.forward(inputs)
	activation1.forward(dense1.outputs)
	dense2.forward(activation1.outputs)
	activation2.forward(dense2.outputs)
	dense3.forward(activation2.outputs)
	activation3.forward(dense3.outputs)
	dense4.forward(activation3.outputs)
	predictions = np.argmax(activation_and_loss.calculate_softmax(dense4.outputs),axis=1)
	accuracy = np.mean(predictions == labels)
	print("{:.2f}".format(accuracy))


print("Number of unreachable objects collected by GC:", n)
input()


epochs = 1000
#batch
BS = 64
steps = int(np.ceil(train_data.shape[0]/BS))

start_time = time.time()
for e in range(epochs):
	for step in range(steps):
		batch_data = train_data[step*BS:(step+1)*BS]
		batch_labels = train_labels[step*BS:(step+1)*BS]
		iterate(batch_data, batch_labels)

	#if(e % 100 == 0):
		#average_CEloss = np.mean(activation_and_loss.celoss_outputs)
		#print(f"epoch:{e} loss:{average_CEloss}")
end_time = time.time()
average_CEloss = np.mean(activation_and_loss.celoss_outputs)
print(f"loss:{average_CEloss}")
		
accuracy(train_data, train_labels)
accuracy(test_data, test_labels)

print("--- %s seconds ---" % (end_time - start_time))
model = np.array([])
model = np.append(model, [dense1, dense2, dense3, dense4])

np.save("model", model)
