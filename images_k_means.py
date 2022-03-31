from sklearn.cluster import KMeans
from glob import glob
import numpy as np
import mahotas
import cv2

def kmean_function(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    ab = lab_image[:, :, 1:3]
    [nrows, ncols] = ab.shape[0:2]
    ab = ab.reshape(nrows * ncols, 2)
    nColors = 2;  # 2 clusters

    kmeans = KMeans(n_clusters=2, random_state=0).fit(ab)
    [cluster_idx, cluster_center] = [kmeans.labels_, kmeans.cluster_centers_]

    ###

    ###
    pixel_labels = cluster_idx.reshape(nrows, ncols);
    pixel_labels = (255 * pixel_labels).astype('uint8')

    gray = pixel_labels

    image_binary = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 0, 0), -1)

    return image_binary


if __name__ == "__main__":
    folder  ='/media/thien/DATA/TaiLieuKy8/ChuyenDe/dataset_da_segmnet/dataset/original'
    folder = '/media/thien/DATA/TaiLieuKy8/ChuyenDe/dataset_chua_segment/rose_chuasegment/rose_segment'
    list_images = glob(folder + '/*.png')
    total_imges = len(list_images)
    for image in list_images:
        print(image)
        image = cv2.imread(image)

        image = mahotas.stretch_rgb(image) 
        image = Gaussian = cv2.GaussianBlur(image, (9, 9), 0)

        image_binary = kmean_function(image)

        cv2.imshow('original', image)
        cv2.imshow('Largest Object', image_binary)

        # cv2.imshow('I',image)
        cv2.waitKey(1)




