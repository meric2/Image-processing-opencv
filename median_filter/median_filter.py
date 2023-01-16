import cv2
import numpy as np

###################Q1###################

print("------------------------Q1------------------------")

def median_filter(img_arr, n):
    padding = n//2
    mid = (n*n)//2
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

    arr = cv2.copyMakeBorder(img_arr, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):#filtering
        for j in range(padding, arr.shape[1]-padding):
            sort = []
            for x in range(0,n):
                for y in range(0,n):
                    sort.append(arr[i-padding+x][j-padding+y])
            sort = np.sort(sort)

            out[i-padding][j-padding] = sort[mid]

    return out


img_array = cv2.imread('noisyImage.jpg', cv2.IMREAD_GRAYSCALE)
my_median = median_filter(img_array, 5)
cv_median = cv2.medianBlur(img_array, 5)

print("Absolute difference between median filters:", (np.abs(my_median-cv_median)).sum())
cv2.imshow("my_median", my_median)
cv2.imshow("cv_median", cv_median)

cv2.waitKey(0)
cv2.destroyAllWindows


###################Q2###################

print("------------------------Q2------------------------")

golden = cv2.imread('lena_grayscale_hq.jpg', cv2.IMREAD_GRAYSCALE)
cv_box = cv2.boxFilter(img_array, 8, (5,5), normalize=True, borderType=cv2.BORDER_CONSTANT)
cv_mean = cv2.GaussianBlur(img_array,(7,7),0)

#cv2.imshow("cv_box", cv_box)
#cv2.imshow("cv_mean", cv_mean)

psnr4 = cv2.PSNR(cv_median, golden)
psnr2 = cv2.PSNR(cv_box, golden)
psnr3 = cv2.PSNR(cv_mean, golden)

print("Psnr of median and noisy:" , psnr4)
print("Psnr of box and noisy:" , psnr2)
print("Psnr of mean and noisy:" , psnr3)

###################Q3###################

print("------------------------Q3------------------------")

def center_weighted_median(img_arr, n):
    padding = n//2
    mid = (n*n+2)//2
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

    arr = cv2.copyMakeBorder(img_arr, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):#filtering
        for j in range(padding, arr.shape[1]-padding):
            sort = []
            for x in range(0,n):
                for y in range(0,n):
                    if x == y:
                        sort.append(arr[i-padding+x][j-padding+y])
                        sort.append(arr[i-padding+x][j-padding+y])    
                    sort.append(arr[i-padding+x][j-padding+y])
            sort = np.sort(sort)

            out[i-padding][j-padding] = sort[mid]

    return out


#1.my_median
#2.cv_box
#3.cv_mean
#4.cv_median
center_med = center_weighted_median(img_array, 5)#5.

cv2.imshow("my_median", my_median)
cv2.imshow("cv_box", cv_box)
cv2.imshow("cv_mean", cv_mean)
cv2.imshow("cv_median", cv_median)
cv2.imshow("center_med", center_med)

cv2.waitKey(0)
cv2.destroyAllWindows

psnr1 = cv2.PSNR(my_median, golden)
psnr5 = cv2.PSNR(center_med, golden)

print("Psnr of my median and noisy:" , psnr1)
print("Psnr of box and noisy:" , psnr2)
print("Psnr of mean and noisy:" , psnr3)
print("Psnr of opencv median and noisy:" , psnr4)
print("Psnr of center weighted median and noisy:" , psnr5)

###################Q4###################

print("------------------------Q4------------------------")

def worsepsnr_center_weighted_median(img_arr, n):
    padding = n//2
    mid = (n*n+2)//2
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

    arr = cv2.copyMakeBorder(img_arr, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):#filtering
        for j in range(padding, arr.shape[1]-padding):
            sort = []
            for x in range(0,n):
                for y in range(0,n):
                    if x == y:
                        sort.append(arr[i-padding+x][j-padding+y])
                        sort.append(arr[i-padding+x][j-padding+y])
                        sort.append(arr[i-padding+x][j-padding+y]) 
                        sort.append(arr[i-padding+x][j-padding+y])
                        sort.append(arr[i-padding+x][j-padding+y])
                        sort.append(arr[i-padding+x][j-padding+y])
                        sort.append(arr[i-padding+x][j-padding+y])
                    sort.append(arr[i-padding+x][j-padding+y])
            sort = np.sort(sort)

            out[i-padding][j-padding] = sort[mid]

    return out

worse = worsepsnr_center_weighted_median(img_array, 7)

cv2.imshow("worse", worse)
cv2.waitKey(0)
cv2.destroyAllWindows

print("Psnr of a different median filter and noisy:", cv2.PSNR(worse, golden))
