from PIL import Image
import random as rd
def imgRandomTrans (img):
	img = img.rotate(rd.randint(-45,45))
	scaleR = rd.uniform(0.77,1.3)
	img = img.resize((int(img.height*scaleR),int(img.width*scaleR)))
	if rd.randint(0,1)==1 :
		img = img.transpose(Image.FLIP_LEFT_RIGHT)

	return img
for i in range(1,7001):
	imageName = str(i).zfill(5)
	path = "/home/ubuntu/caffe/examples/images/joey/"+imageName+".jpg"
	img = Image.open(path)
	img = imgRandomTrans(img)
	img.save("/home/ubuntu/hdd/imageRotate/flip_img/"+str(10000+i)+".jpg")
	img = imgRandomTrans(img)
	img.save("/home/ubuntu/hdd/imageRotate/flip_img/"+str(20000+i)+".jpg")
	img = imgRandomTrans(img)
	img.save("/home/ubuntu/hdd/imageRotate/flip_img/"+str(30000+i)+".jpg")
	img = imgRandomTrans(img)
	img.save("/home/ubuntu/hdd/imageRotate/flip_img/"+str(40000+i)+".jpg")
	img = imgRandomTrans(img)
	img.save("/home/ubuntu/hdd/imageRotate/flip_img/"+str(50000+i)+".jpg")
