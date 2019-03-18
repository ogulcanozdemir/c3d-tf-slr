import numpy as np
import cv2
import os


def list_length(train_lst):
	with open(train_lst) as f:
		for i, l in enumerate(f):
			pass
		return i + 1


def compute_volume_mean(train_lst, num_frames=16, new_w_h_size=112):
	mean = np.zeros((num_frames, new_w_h_size, new_w_h_size, 3))
	count = 0
	
	with open(train_lst) as f:
		length = list_length(train_lst)
		
		for idx, line in enumerate(f):
			print('Reading line {}/{}'.format(idx, length))
			vid_path = line.split()[0]
			stack_frames = []

			for i in range(1, num_frames+1):
				img = cv2.imread(os.path.join(vid_path, "{:05}.jpg".format(i)))
				img = cv2.resize(img, (new_w_h_size, new_w_h_size))
				stack_frames.append(img)

			stack_frames = np.array(stack_frames)
			mean += stack_frames
			count += 1

	mean /= float(count)
	print(mean)
	return mean


if __name__ == '__main__':
	mean_general_16 = compute_volume_mean('list/general/train.list')
	np.save('crop_mean_16.npy', mean_general_16)
