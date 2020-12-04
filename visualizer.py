import cv2
import numpy as np
from math import floor, pi
from sklearn.cluster import MeanShift, estimate_bandwidth


class BloodCellVisualizer:
    def __init__(self, bloodcelldetector, convnet):
        self.detector = bloodcelldetector
        self.convnet = convnet
        self.marker_size = 40

    def highlight_cells(self, img, labels=None, score_thresh=0.95, show_scores=False, make_zoom=False):
        self.detector.img = img.copy()
        self.detector.find_cells()

        result_img = img.copy()
        colors = [(0, 255, 255), (0, 0, 255), (0, 255, 0)]

        for i, cell_coordinates in enumerate(self.detector.overlays):
            label_index, scores = self.convnet.predict(self.detector.cells_images[i], with_scores=True)

            if max(scores) < score_thresh:
                continue

            if labels is not None and self.convnet.labels[label_index] not in labels:
                continue

            cv2.rectangle(result_img,
                          cell_coordinates,
                          (cell_coordinates[0] + self.marker_size, cell_coordinates[1] + self.marker_size),
                          colors[label_index], 2)

            if show_scores:
                x = cell_coordinates[0] - (self.marker_size - 10)
                for j in range(3):
                    cv2.putText(result_img, str(scores[j]),
                                (x, cell_coordinates[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colors[j], 1)
                    x += self.marker_size

        if make_zoom:
            center, radius = self.zoom_neutrophils()
            if center is not None:
                cv2.circle(img=result_img,
                           center=center,
                           radius=radius,
                           color=colors[2],
                           thickness=3)

        return result_img

    def zoom_neutrophils(self):

        img_size = min(self.detector.img.shape[0:2])

        neutrophils_objects = []

        # get neutrophils objects
        for i, cell_coordinates in enumerate(self.detector.overlays):
            prediction = self.convnet.predict(self.detector.cells_images[i])
            if prediction == 2:
                neutrophils_objects.append(cell_coordinates)

        if len(neutrophils_objects) == 0:
            return None, None

        quantile = 0.4 / (2 ** min(floor(img_size / 1000.0), 3))
        bandwidth = estimate_bandwidth(neutrophils_objects, quantile=quantile)
        if bandwidth <= 0:
            return None, None
        else:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(neutrophils_objects)

            labels = ms.predict(neutrophils_objects)
            clusters = []
            for label in np.unique(labels):
                clusters.append([neutrophils_objects[i] for i, lbl in enumerate(labels) if lbl == label])

            def density_function(x, objects, ms_density, convnet_density):
                current_index = objects.index(x)
                bias_density = int(convnet_density.img_size / 2)
                center_density = (int(ms_density.cluster_centers_[current_index][0]) + bias_density,
                                  int(ms_density.cluster_centers_[current_index][1]) + bias_density)
                radius_density = int(max([np.linalg.norm(np.array(coord) - np.array(center_density)) for coord in x]))
                area = pi * radius_density ** 2
                density = len(x) / area if len(x) > 1 else 0

                return density

            object_in_max_clusters = max(clusters, key=lambda x: density_function(x, clusters, ms, self.convnet))
            index_of_maximum = clusters.index(object_in_max_clusters)
            bias = int(self.convnet.img_size / 2)

            # get optimal center and radius of cluster
            center = (int(ms.cluster_centers_[index_of_maximum][0]) + bias,
                      int(ms.cluster_centers_[index_of_maximum][1]) + bias)
            radius = int(max([np.linalg.norm(np.array(coord) - np.array(center)) for coord in object_in_max_clusters]))
            return center, radius + bias
