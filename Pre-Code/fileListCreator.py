

def writeFile(filename, sourcePath):
	
	f = open(filename, "w")
	for i in range(1,7001):
		d = sourcePath + str(i).zfill(5) + ".jpg 0\n"
		f.write(d)
	for i in range(10001,17001):
		d = sourcePath + str(i).zfill(5) + ".jpg 0\n"
		f.write(d)
	for i in range(20001,27001):
		d = sourcePath + str(i).zfill(5) + ".jpg 0\n"
		f.write(d)
	for i in range(30001,37001):
		d = sourcePath + str(i).zfill(5) + ".jpg 0\n"
		f.write(d)
	f.close()
	return

def main():
	writeFile("train_file_list_add_rot.txt", "/home/ubuntu/caffe/examples/images/joey/")
	#writeFile("test_file_list_withoutL.txt", "/home/ubuntu/caffe/examples/images/val/")
if __name__ == '__main__':
	main()