from numpy import genfromtxt
import numpy as np

def writeFile(filename, sourcePath):
	
	f = open(filename, "w")
	for i in range(1,7001):
		d = sourcePath + str(i).zfill(5) + ".jpg 0\n"
		f.write(d)
	f.close()
	return


my_data = genfromtxt('../train.csv', delimiter=',')
clean_data = my_data[1:,1]
print (clean_data.shape)
suma = 0;
f = open("Addtional_train", "w")


i = 7
condition = clean_data == i 
temp = np.where(condition)[0]
print (temp)
for i in range(20):
	for line in temp:
		d = "/home/ubuntu/caffe/examples/images/joey/" + str(line).zfill(5) + ".jpg 0\n"
		f.write(d)


i = 6
condition = clean_data == i 
temp = np.where(condition)[0]
print (temp)
for i in range(4):
	for line in temp:
		d = "/home/ubuntu/caffe/examples/images/joey/" + str(line).zfill(5) + ".jpg 0\n"
		f.write(d)

i = 8
condition = clean_data == i 
temp = np.where(condition)[0]
print (temp)
for i in range(7):
	for line in temp:
		d = "/home/ubuntu/caffe/examples/images/joey/" + str(line).zfill(5) + ".jpg 0\n"
		f.write(d)


f.close()

f = open("Addtional_label", "w")
for i in range(19*20):
	d = "7\n"
	f.write(d)
for i in range(91*4):
	d = "6\n"
	f.write(d)
for i in range(48*7):
	d = "8\n"
	f.write(d)
f.close()

# for i in range(1,9):
# 	a = np.sum(clean_data == i)
# 	suma += a
# 	print ("class"+str(i)+":"+str(a))
# print ("Sum is "+str(suma))
