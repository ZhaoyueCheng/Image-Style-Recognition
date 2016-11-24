from numpy import genfromtxt
import numpy as np

my_data = genfromtxt('../train.csv', delimiter=',')
clean_data = my_data[1:,1]
print (clean_data.shape)
suma = 0;
for i in range(1,9):
	a = np.sum(clean_data == i)
	suma += a
	print ("class"+str(i)+":"+str(a))
print ("Sum is "+str(suma))
