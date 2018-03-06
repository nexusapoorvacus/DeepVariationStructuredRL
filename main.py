from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import VGDataset

import torch
import torch.nn as nn
import argparse
import json

def train(data_loader):
	for progress, images in enumerate(data_loader):
		print(progress)

if __name__=='__main__':
	# flags
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_data", type=str,
				default="data/train_data.json",
				help='Location of the file containing train data samples')
	parser.add_argument("--images_dir", type=str,
				default="data/images/",
				help="Location of Visual Genome images")
	parser.add_argument("--train", help="trains model", action="store_true")
	parser.add_argument("--test", help="evaluates model", action="store_true")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size to use")
	parser.add_argument("--num_workers", type=int, default=4, help="number of threads")
	args = parser.parse_args()

	# load train data samples
	if args.train:
		train_data_samples = json.load(open(args.train_data))
		train_dataset = VGDataset(train_data_samples, args.images_dir)
		train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
									shuffle=True, num_workers=args.num_workers)
		train(train_data_loader)
