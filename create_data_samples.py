import json
import argparse
import os

IMAGE_DIR = "data/images/"
OUTPUT_DATA_SAMPLE_FILE = "data/data_samples.json"
NUM_IMAGES = 10

def create_data_sample_file():
	file_to_write = open(OUTPUT_DATA_SAMPLE_FILE, "w")
	images = []
	for im_name in os.listdir(IMAGE_DIR)[:NUM_IMAGES]:
		images.append(im_name)
	file_to_write.write(str(images))

def main():
	print("Creating data sample file for " + str(NUM_IMAGES)  + " images...")
	create_data_sample_file()
	print("Done!")

if __name__ == "__main__":
	main()
