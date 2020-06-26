from datetime import datetime
import time, cv2, itertools, os, json, glob
import numpy as np
import pandas as pd

from urllib.request import urlopen

def url_to_image(url, category, index, size):
	req = urlopen(url)
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	category = category.replace(" ", "_")
	category_folder_train = "./data/train/{}".format(category)
	category_folder_validation = "./data/validation/{}".format(category)
	category_folder_test = "./data/test/{}".format(category)
	if index < int(0.8 * size):
		if not os.path.exists(category_folder_train):
			os.makedirs(category_folder_train)
		file_name = category_folder_train + "/{}_{}.jpg".format(category, index)
		print(file_name)
	elif index < int(0.9 * size) and index > int(0.8 * size):
		if not os.path.exists(category_folder_validation):
			os.makedirs(category_folder_validation)
		file_name = category_folder_validation + "/{}_{}.jpg".format(category, index)
		print(file_name)
	else:
		if not os.path.exists(category_folder_test):
			os.makedirs(category_folder_test)
		file_name = category_folder_test + "/{}_{}.jpg".format(category, index)
		print(file_name)
	cv2.imwrite(file_name, img)
	''' End Function '''

def main():
	image_categories = [
		'Women',
		'Children',
		'Animals',
		'Uniforms',
		'Buildings',
		'Street scene',
		'Vehicles',
		'Signs',
		'Weapons',
		'Railroad cars',
		'Nazi symbols',
		'Gravestones',
		'Barbed wire fences',
		'Corpses',
		'German soldiers',
		'Armband',
		'Snow',
		'Carts',
	]
	image_categories = sorted(image_categories)
	all_csvs = glob.glob('./csv/*.csv')
	all_csvs = sorted(all_csvs)
	image_dictionary = dict(zip(image_categories, all_csvs))
	for category in image_dictionary:
		category_dataset = pd.read_csv(image_dictionary[category])
		category_dataset.sort_values("url", inplace=True)
		category_dataset.drop_duplicates(subset="url", inplace=True)

		counter = 0
		for index, row in category_dataset.iterrows():
			url_to_image(row['url'], category, counter, category_dataset.shape[0])
			counter += 1


if __name__ == "__main__":
	main()