import cv2
import numpy as np


###################Q1###################

print("###################Q1###################")

def adaptive_mean_filter(img_arr):
    img8 = np.uint8(img_arr)
    normalized = cv2.normalize(img8, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    padding = 5//2#Sxy=5x5
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    
    arr = cv2.copyMakeBorder(normalized, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            a = []
            for x in range(0,5):
                for y in range(0,5):
                    a.append(arr[i-padding+x][j-padding+y])
            var = np.var(a)
            avr = np.average(a)

            out[i-padding][j-padding] = arr[i][j]-((0.004/var)*(arr[i][j]-avr))
    
    k = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.uint8(k)


img_array = cv2.imread("noisyImage_Gaussian.jpg", cv2.IMREAD_GRAYSCALE)

output_1_1 = adaptive_mean_filter(img_array)
output_1_2 = cv2.boxFilter(img_array, 8, (5,5), normalize=True, borderType=cv2.BORDER_CONSTANT)
output_1_3 = cv2.GaussianBlur(img_array,(5,5),0)

clean = cv2.imread("lena_grayscale_hq.jpg", cv2.IMREAD_GRAYSCALE)

psnr_1_1 = cv2.PSNR(output_1_1, clean)
psnr_1_2 = cv2.PSNR(output_1_2, clean)
psnr_1_3 = cv2.PSNR(output_1_3, clean)

print("PSNR of adaptive mean filter:", psnr_1_1)
print("PSNR of box filter:", psnr_1_2)
print("PSNR of Gaussian filter:", psnr_1_3)

cv2.imshow("output_1_1 psnr "+str(psnr_1_1), output_1_1)
cv2.imshow("output_1_2 psnr "+str(psnr_1_2), output_1_2)
cv2.imshow("output_1_3 psnr "+str(psnr_1_3), output_1_3)
cv2.waitKey(0)
cv2.destroyAllWindows

###################Q2###################

print("###################Q2###################")

def adaptive_median_filter(img_arr):
    S = [3,5,7]
    S_ind = 0
    padding = S[-1]//2#Smax
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

    img8 = np.uint8(img_arr)
    normalized = cv2.normalize(img8, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    arr = cv2.copyMakeBorder(img_arr, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            while(True):
                a = []
                for x in range(0,S[S_ind]):
                    for y in range(0,S[S_ind]):
                        a.append(arr[i-padding+x][j-padding+y])
                z_min = np.min(a)
                z_max = np.max(a)
                z_med = np.median(a)
                z_xy = arr[i][j]

                if z_min<z_med and z_med<z_max:#B
                    if z_min<z_xy and z_xy<z_max:
                        out[i-padding][j-padding] = z_xy
                        break
                    else:
                        out[i-padding][j-padding] = z_med
                        break
                else:#A
                    if S_ind < 2:
                        S_ind += 1
                    else:
                        S_ind = 0
                if S_ind == 2:
                    out[i-padding][j-padding] = z_med
                    break
    k = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.uint8(k)
    

def center_weighted_median(img_arr, n):
    padding = n//2
    mid = (n*n+2)//2
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

    arr = cv2.copyMakeBorder(img_arr, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            sort = []
            for x in range(0,n):
                for y in range(0,n): 
                    sort.append(arr[i-padding+x][j-padding+y])
            for k in range(0,n-1):
                sort.append(arr[i][j]) 
            sort = np.sort(sort)

            out[i-padding][j-padding] = sort[mid]

    return out


img_sp = cv2.imread("noisyImage_SaltPepper.jpg", cv2.IMREAD_GRAYSCALE)
#img_sp = cv2.imread("noisyImage.jpg", cv2.IMREAD_GRAYSCALE)

output_2_1 = adaptive_median_filter(img_sp)
output_2_2 = cv2.medianBlur(img_sp, 3)#salt
output_2_3 = cv2.medianBlur(img_sp, 5)
output_2_4 = cv2.medianBlur(img_sp, 7)
output_2_5 = center_weighted_median(img_sp, 3)#pepper
output_2_6 = center_weighted_median(img_sp, 5)
output_2_7 = center_weighted_median(img_sp, 7)

psnr_2_1 = cv2.PSNR(output_2_1, clean)
psnr_2_2 = cv2.PSNR(output_2_2, clean)
psnr_2_3 = cv2.PSNR(output_2_3, clean)
psnr_2_4 = cv2.PSNR(output_2_4, clean)
psnr_2_5 = cv2.PSNR(output_2_5, clean)
psnr_2_6 = cv2.PSNR(output_2_6, clean)
psnr_2_7 = cv2.PSNR(output_2_7, clean)

print("PSNR of adaptive median filter:", psnr_2_1)
print("PSNR of median filter(3x3):", psnr_2_2)
print("PSNR of median filter(5x5):", psnr_2_3)
print("PSNR of median filter(7x7):", psnr_2_4)
print("PSNR of center weighted median filter(3x3):", psnr_2_5)
print("PSNR of center weighted median filter(5x5):", psnr_2_6)
print("PSNR of center weighted median filter(7x7):", psnr_2_7)

cv2.imshow("output_2_1 psnr "+str(psnr_2_1), output_2_1)
cv2.imshow("output_2_2 psnr "+str(psnr_2_2), output_2_2)
cv2.imshow("output_2_3 psnr "+str(psnr_2_3), output_2_3)
cv2.imshow("output_2_4 psnr "+str(psnr_2_4), output_2_4)
cv2.imshow("output_2_5 psnr "+str(psnr_2_5), output_2_5)
cv2.imshow("output_2_6 psnr "+str(psnr_2_6), output_2_6)
cv2.imshow("output_2_7 psnr "+str(psnr_2_7), output_2_7)
cv2.waitKey(0)
cv2.destroyAllWindows