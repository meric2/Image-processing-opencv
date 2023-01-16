import cv2
import numpy as np


def clahe(img_arr, n, t):
    padding = n//2
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    
    arr = cv2.copyMakeBorder(img_arr, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            hist = np.zeros((256))
            for x in range(0,n):
                for y in range(0,n):
                    hist[arr[i-padding+x][j-padding+y]] += 1

            sum = 0
            for k in range(len(hist)):
                    if hist[k] > t:
                        sum += hist[k]-t
                        hist[k] = t
            lim_hist = hist + (sum/len(hist))

            cdf = np.zeros(256)
            cdf[0] = lim_hist[0]
            cdf = np.cumsum(lim_hist)
            out[i-padding][j-padding] = np.round(cdf[arr[i][j]])

    out = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.uint8(out)



img_array = cv2.imread("test6.jpg", cv2.IMREAD_GRAYSCALE)

out1 = clahe(img_array, 16, 4)
out2 = cv2.equalizeHist(img_array)

#clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
#cvclahe = clahe.apply(img_array)

cv2.imshow("My CLAHE", out1)
cv2.imshow("Histogram Equalization", out2)
cv2.waitKey(0)
cv2.destroyAllWindows