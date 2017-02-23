from numpy import genfromtxt

predict = genfromtxt("res152-DA.csv",delimiter=',',dtype=int)
label = genfromtxt("Dangerous_all.csv",delimiter=',',dtype=int)

predict = predict[1:,1]
label = label[1:,1]


predict = predict[970:]
label = label[970:]
print predict.shape
print label.shape

con = predict == label

print float(sum(con))/float(label.shape[0])
