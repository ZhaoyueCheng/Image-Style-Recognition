from numpy import genfromtxt
import numpy as np

my_data = genfromtxt('../train.csv', delimiter=',')
clean_data = my_data[1:,1]
suma = 0;
for i in range(8):
	a = np.sum(my_data == i+1)
	suma += a
	print (a)
print ("Sum is "+str(suma))
