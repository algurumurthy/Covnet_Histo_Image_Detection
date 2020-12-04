import cv2
import numpy as np
import os
from scipy import misc


class BloodCellDetector:
    """
    Detect blood cells in an image
    """
    def __init__(self, img_path, img_name):
        self.img_path = img_path
        self.img_name = img_name
        self.img, self.cells_images, self.overlays = None, None, None

    @staticmethod
    def get_contours(img, kernel_size, lower_purple, upper_purple):
        """Get contouts of the blood cells"""
        imgc = img.copy()
        hls = cv2.cvtColor(imgc, cv2.COLOR_BGR2HLS)
        hls = cv2.GaussianBlur(hls, (kernel_size, kernel_size), 10)
        lower_purple = np.array(lower_purple)
        upper_purple = np.array(upper_purple)
        mask = cv2.inRange(hls, lower_purple, upper_purple)
        res = cv2.bitwise_and(imgc, imgc, mask=mask)
        res[np.where((res == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        img_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 2)
        edges = cv2.Canny(thresh.copy(), 1, 2)
        blur = cv2.GaussianBlur(edges, (1, 1), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_64F, 1, 1, 3)
        _, contours, hierarchy = cv2.findContours(laplacian.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy

    @staticmethod
    def centers(contours):
        """
        Find centers of all the contours
        :param contours: List of contours
        :return: List of tuples-points (x, y)
        """
        centers = []
        for cnt in contours:
            moment = cv2.moments(cnt)
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            centers.append((cx, cy))
        return centers

    @staticmethod
    def is_unique_contour(marks, pt, min_dist):
        for coordinates in marks:
            dist = np.linalg.norm(np.array(coordinates) - np.array(pt))
            if dist < min_dist:
                return False
        return True

    def load_image(self):
        self.img = cv2.imread(os.path.join(self.img_path, self.img_name))
        print('image loaded')

    def find_cells(self,
                   kernel_size=5,
                   lower_color=None, upper_color=None,
                   min_area=20, max_area=1000,
                   img_size=40, resize_img=50,
                   min_dist=20):
        if upper_color is None:
            upper_color = [150, 80, 220]
        if lower_color is None:
            lower_color = [120, 10, 100]
        imgc = self.img.copy()
        nucleus, hierarchy = self.get_contours(imgc, kernel_size, lower_color, upper_color)
        nucleus = [cv2.convexHull(cnt) for i, cnt in enumerate(nucleus) if
                   max_area > cv2.contourArea(cnt) > min_area and hierarchy[0][i][3] == -1]
        keypoints = self.centers(nucleus)
        self.cells_images, self.overlays = [], []
        uniques = []

        for k in keypoints:
            if self.is_unique_contour(uniques, k, min_dist):
                uniques.append(k)

        for cells_coordinates in uniques:
            x = int(cells_coordinates[0]) - int((img_size / 2))
            y = int(cells_coordinates[1]) - int((img_size / 2))
            roi = self.img[y:y + img_size, x:x + img_size]
            if roi.size:
                self.overlays.append((x, y))
                img_file = misc.imresize(arr=roi, size=(resize_img, resize_img, 3))
                img_arr = np.asarray(img_file)
                self.cells_images.append(img_arr)

        self.cells_images = np.asarray(self.cells_images)
