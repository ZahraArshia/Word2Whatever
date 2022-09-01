import glob
import cv2
import numpy as np # linear algebra
import time
import skimage.measure
import matplotlib.pyplot as plt

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err
def compare_images(imageA, imageB):

    m = mse(imageA, imageB)
    s = skimage.measure.compare_ssim(imageA, imageB,multichannel=True)
    print("MSE: %.2f, SSIM: %.2f" % (m, s))
    return (s)
result1 =[]
de = []
counter_result = 0
f_res = []
d = 0
c = 0
start = time.time()
all_result =[]
res1=[]
class_names =[]
result = []
test_filenames = glob.glob("test/" + "*.jpg")
for test_index in test_filenames:
    original = cv2.imread(test_index)
    # print(test_index)
    test_class = test_index.split('.jpg')[0].split(',')[2]
    # print(test_class)
    # test_class = tfilen.split('')[0]
    for index in range(-861,2055,15):
        yclass = "c" + str(index)
        filenames = glob.glob("data/"+ str(yclass)+ "/*.jpg")
        filenames.sort()
        if len(filenames) != 0:
            label_class = filenames[0].split('.jpg')[0].split(',')[2]
            # print(filen)
            # label_class =filen.split('\\')[0]
            print('Start in this Class:' + str(label_class))
        for image_path in filenames:
                img = cv2.imread(image_path)
                images = ("Original",original), ("Train", img)
                rs = compare_images(original, img)
                # if label_class == test_class and  rs >= 0.90:
                #     counter_result = counter_result + 1
                #     print(label_class)
                #     print('and')
                #     print(test_class)
                #     print("=================================")
                #     print(counter_result)
                result1.append(rs)
                res1.append((rs,label_class,test_class))
                # de.append(label_class)
        # filenames = []

                # print("Max SSIM Is: (SSIM,Folder,Image Name)")
        # print(res1)
    f_res.append(max(result1))   # max har daste
    maxm = max(result1)
    result1 = []
# maxm = max(f_res)
    # maxm.index()
all_result.append(maxm)
print(maxm)
# print("Max SSIM in this Class : " + str(max(result1)))
print('End of All Classes')
print('===========================================')

# print(str(f_res))
# print(all_result)
# print('===================')
# print(result1)

# print(f_res)
# print("----------------------")
for ind in f_res:
    # if ind == 1.00 or ind == -1 or ind == 1 or ind == float(1.00) or ind == "1":
    if ind >= 0.80:
        counter_result = counter_result + 1
        print(ind)
        print("=================================")
        print(counter_result)
        print(len(test_filenames))
final_res = float(counter_result/len(test_filenames))*100
print("Accuracy is:")
print(final_res)

#
#
# for ind in result1:
#     if ind == 1.0:
#         counter_result = counter_result + 1
#         print(ind)
#         print("=================================")
#         print(counter_result)
# final_res = (counter_result/len(result1))
# print(final_res)
# print("result:")
# print(res1)