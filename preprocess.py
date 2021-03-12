import numpy as np
import cv2 

def show_image(image, label = "window"):
	cv2.namedWindow(label)
	cv2.imshow(label, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)

def labels_to_logits(labels):
	labels_logits = np.array([], dtype="int8")
	labels_uniques = np.unique(labels)
	for label in labels:
		labels_logits = np.append(labels_logits, np.where(labels_uniques == label))
	return labels_logits.reshape(len(labels))
	
def augmentImage(image, *args):
	arguments = list(*args)
	arguments = [x.lower() for x in arguments[1:]]
	augmentedImages = []
	if("verticalflip" in arguments):
		augmentedImages.append(cv2.flip(image, 0))
		arguments.remove("verticalflip")
	if("horizontalflip" in arguments):
		augmentedImages.append(cv2.flip(image, 1))
		arguments.remove("horizontalflip")
	if("rotation" in arguments):
		augmentedImages.append(cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE))
		augmentedImages.append(cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE))
		arguments.remove("rotation")

	if(len(arguments)):
		unknownArgumentsException(arguments)

	return np.array(augmentedImages)

def unknownArgumentsException(listArgs):
	error = "Unknown Arguments: " + " ".join(listArgs)
	raise Exception(error)
	
	
def shuffle_data_labels(data, labels):
	shuffled_indices = np.random.permutation(data.shape[0])
	return data[shuffled_indices], labels[shuffled_indices]
	
def augmentImages(data, labels, *args):
	
	newData = []
	newLabelsLogits = []

	try:
		for imageIndex in range(len(data)):
			image = data[imageIndex]
			for augmented in augmentImage(image, args):
				newData.append(augmented)
				newLabelsLogits.append(labels[imageIndex])
	except e:
		print(e) 
			
	
	newData = np.append(data, newData ,axis = 0)
	newLabelsLogits = np.append(labels, newLabelsLogits, axis = 0)

	return shuffle_data_labels(newData , newLabelsLogits)

def split_train_test(data, labels, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]
   
   
