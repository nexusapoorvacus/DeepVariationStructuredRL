import json
import argparse

IMAGE_DIR = "data/images/"
OUTPUT_DATA_SAMPLE_FILE = "data/data_samples.json"
NUM_IMAGES = 10

def create_data_sample_file():
	image_files = os.scandir(IMAGE_DIR)
	file_to_write = open(OUTPUT_DATA_SAMPLE_FILE, "w")
	images = []
	for im in image_files:
		images.append(im.name)
	file_to_write.write(str(images))

def main():
	print("Creating data sample file for " + str(NUM_IMAGES)  + " images...")
	create_data_sample_file()
	print("Done!")
