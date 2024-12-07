"""
Project: EXTRACT OBJECT FROM THE BACKGROUND
Group: TRAN CHI CUONG, DAO DUY NGU, LE VAN THIEN, NGUYEN VU HOAI DUY, HO THANH LONG
The programing created by DAO DUY NGU, LE VAN THIEN
date: 22/03/2022
"""
from glob import glob

try:
    import cv2
    import numpy as np

except:
    raise Exception('cv2 or numpy packages not found. Installation: $ pip install opencv-python or numpy')


class Kmeans:
    def __init__(self, k_cluster):
        self.k_cluster = k_cluster

    def extract_object(self, image):
        """
        function extract object from background use kmeans, find contours, flood fill
        :param image: image color channel BGR
        :return: image and mask after extract and crop
        """
        # convert channel BGR->RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # setup vector
        vectorized = image_rgb.reshape((-1, 3))
        vectorized = np.float32(vectorized)
        # setup criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # setup attempts
        attempts = 10
        # run Kmeans
        ret, label, centers = cv2.kmeans(vectorized, self.k_cluster, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        # reshape label about size image binary
        image_label = label.reshape(image.shape[:2])
        # get size height, width of image
        height, width, _ = image.shape
        # find centroid and get thresh = min(height, width) / 4
        cx, cy = width // 2, height // 2
        thresh = min(width, height) // 4
        # get image_check centroid
        image_check = image_label[cy - thresh:cy + thresh, cx - thresh:cx + thresh]
        # find numbers max cluster
        num_cluster_max = np.sum(image_check == 0)
        k_max = 0
        for k in range(1, self.k_cluster):
            num_cluster = np.sum(image_check == k)
            if num_cluster_max < num_cluster:
                num_cluster_max = num_cluster
                k_max = k
        # create mask follow numbers of max cluster
        mask_image = np.uint8(np.copy(image_label))
        mask_image[image_label == k_max] = 255
        mask_image[image_label != k_max] = 0
        image_contours, bbox = self.find_contours(mask_image)
        mask_image = self.flood_fill(image_contours)
        x1, y1, w, h = bbox
        image = cv2.bitwise_and(image, image, mask=mask_image)
        masked = mask_image[y1:y1 + h, x1:x1 + w]
        image = image[y1:y1 + h, x1:x1 + w]
        return image, masked

    @staticmethod
    def flood_fill(image_binary):
        """
        function flood fill for image binary
        :param image_binary: image binary
        :return: image after flood fill
        """
        h, w = image_binary.shape
        mask_zeros = np.zeros((h + 2, w + 2), dtype=np.uint8)
        holes = cv2.floodFill(image_binary.copy(), mask_zeros, (0, 0), 255)[1]
        holes = ~holes
        image_binary[holes == 255] = 255
        return image_binary

    @staticmethod
    def find_contours(image_binary):
        """
        function find contours max and bounding box of image binary
        :param image_binary: image binary
        :return: mask, bbox
        """
        # find edges with canny
        edges = cv2.Canny(image_binary, 100, 200)
        # dilate edges
        edges = cv2.dilate(edges, None)
        # find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # find contours have len max
        value_max = 0
        idx_max = 0
        for idx in range(len(contours)):
            count, _, _ = contours[idx].shape
            if value_max < count:
                value_max = count
                idx_max = idx
        # get contours max
        cnt_max = contours[idx_max]
        bbox = cv2.boundingRect(cnt_max)
        # create mask
        mask_zeros = np.zeros(edges.shape, np.uint8)
        # draw mask with contours
        masked = cv2.drawContours(mask_zeros, [cnt_max], -1, 255, -1)
        return masked, bbox

    @staticmethod
    def flood_fill_1(image):
        img_height, img_width = image.shape
        # find center (21x21 pixel) region of image frame
        center_half = 10  # half-width of 21 is 21/2-1
        center = image[(img_height // 2 - center_half):(img_height // 2 + center_half),
                 (img_width // 2 - center_half):(img_width // 2 + center_half)]

        # find median depth value of center region
        center = np.reshape(center, np.prod(center.shape))
        med_val = np.median(np.reshape(center, np.prod(center.shape)))
        # try this instead:
        abs_depth_dev = 14
        frame = np.where(abs(image - med_val) <= abs_depth_dev, 128, 0).astype(np.uint8)

        # morphological
        kernel = np.ones((3, 3), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        # connected component
        small_kernel = 3
        frame[(img_height // 2 - small_kernel):(img_height // 2 + small_kernel),
        (img_width // 2 - small_kernel):(img_width // 2 + small_kernel)] = 128

        mask_zeros = np.zeros((img_height + 2, img_width + 2), np.uint8)
        flood = frame.copy()
        cv2.floodFill(flood, mask_zeros, (img_width // 2, img_height // 2), 255, flags=4 | (255 << 8))
        ret, flooded = cv2.threshold(flood, 129, 255, cv2.THRESH_BINARY)
        return flooded


if __name__ == '__main__':
    # read image
    path_folder = 'D:/CD_KTMT/Flower-17-dataset-master/Dataset_DDN/Hibiscus/JPG'
    list_path = glob(path_folder + '/*.jpg')
    extract = Kmeans(k_cluster=3)
    for path in list_path:
        img = cv2.imread(path)
        cv2.imshow('image', img)
        img, mask = extract.extract_object(img)
        cv2.imshow('mask', mask)
        cv2.imshow('object', img)
        cv2.waitKey(0)
