# import all necessary libraries here
import numpy as np
import cv2
import extractor
from scipy import ndimage as ndimg
import utils
import math


class VehicleDetector(object):
    '''
    VehicleDetector class. Contains methods to detect vehicles in an image and mark bounding boxes over the detections.
    '''

    def __init__(self, clf, params, scaler):
        '''
        Initialize the object.
        :param clf: the classifier to be used
        :param params: the params for the feature extraction as a dictionary
        :param scaler: the scaler used for scaling the features
        '''

        # initialize all prperties here
        self.clf = clf
        self.params = params
        self.scaler = scaler
        self.processed_frames = 0
        self.avg_frames = 10
        self.heatmaps = []
        self.boxes = []

    def get_all_windows(self, img, x_bounds=[None, None], y_bounds=[None, None], window_dim=(64, 64),
                        overlap=(0.5, 0.5)):
        '''
        Method to get all possible windows for searching in an image
        :param img: the input image
        :param x_bounds: the start and stop value in the x dimension
        :param y_bounds: the start and stop value in the y dimension
        :param window_dim: the search window size
        :param overlap: the percentage of overlap between windows
        :return:a list of window dimensions according to the input
        '''

        image_size = img.shape

        # if x and/or y start/stop positions not defined, set to image size
        x_bounds = (
            x_bounds[0] if x_bounds[0] is not None else 0,
            x_bounds[1] if x_bounds[1] is not None else image_size[1],
        )
        y_bounds = (
            y_bounds[0] if y_bounds[0] is not None else 0,
            y_bounds[1] if y_bounds[1] is not None else image_size[0],
        )

        # calculate the span of the region to be searched
        x_span = x_bounds[1] - x_bounds[0]
        y_span = y_bounds[1] - y_bounds[0]

        # calculate the number of pixels per step in x/y
        x_pix_step = np.int(window_dim[0] * (1 - overlap[0]))
        y_pix_step = np.int(window_dim[1] * (1 - overlap[1]))

        # calculate the number of windows in x/y
        nx_buffer = np.int(window_dim[0] * (overlap[0]))
        ny_buffer = np.int(window_dim[1] * (overlap[1]))
        nx_windows = np.int((x_span - nx_buffer) / x_pix_step)
        ny_windows = np.int((y_span - ny_buffer) / y_pix_step)

        window_list = []

        # calculate the window dimensions for each window and append to the list
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                startx = xs * x_pix_step + x_bounds[0]
                endx = startx + window_dim[0]
                starty = ys * y_pix_step + y_bounds[0]
                endy = starty + window_dim[1]

                window_list.append(((startx, starty), (endx, endy)))

        return window_list

    def _get_feature(self, img):
        '''
        Method to extract the feature of the image according to the train params
        :param img: the input image
        :return: the feature vector of the image
        '''
        return extractor.get_feature(img, cspace=self.params['cspace'],
                                     spatial_size=self.params['spatial_size'], hist_bins=self.params['hist_bins'],
                                     orient=self.params['hog_orient'], pix_per_cell=self.params['hog_pix_per_cell'],
                                     cell_per_block=self.params['hog_cell_per_block'],
                                     hog_channels=self.params['hog_channel'], spatial_feat=self.params['incl_spat'],
                                     hist_feat=self.params['incl_hist'], hog_feat=self.params['incl_hog'])

    def get_raw_predicted_windows(self, img, windows):
        '''
        Method to get the prediction from the classifier for the generated windows
        :param img: the input image
        :param windows: a list of windows to predict
        :return: a list of windows that is classified as a car
        '''
        pred_windows = []

        for window in windows:

            # get the feature for each of the patch formed by the window
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self._get_feature(test_img)
            features_scaled = self.scaler.transform(np.array(features).reshape(1, -1))

            # predict a value using the classifier. Threshold the prediction based on the confidence
            prediction = self.clf.predict(features_scaled)
            conf = self.clf.decision_function(features_scaled)
            if prediction == 1 and conf > 0.4:
                pred_windows.append(window)

        # return the list of confident windows
        return pred_windows

    def get_heatmap(self, img, bboxes):
        '''
        Method to generate a heatmap from the windows
        :param img: the input image
        :param bboxes: the bounding boxes
        :return: a heatmap of the predicted windows
        '''

        # create a blank image
        heatmap = np.zeros(img.shape[:2])

        # iterate through list of windows
        for box in bboxes:
            # set the pixel value for the positive windows
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # return heatmap
        return heatmap

    def get_average_heatmap(self, heatmap):
        '''
        Method to add all the heatmaps together
        :param heatmap: the recent heatmap
        :return: a combined heatmap of recent frames
        '''

        # reset the heatmaps to empty if the frame count is reached
        if len(self.heatmaps) % self.avg_frames == 0:
            self.heatmaps = []

        # add the new heatmap to the recent list
        self.heatmaps.append(heatmap)

        # combine all heatmaps and return it
        all_heatmaps = np.zeros_like(heatmap)
        for hmap in self.heatmaps:
            all_heatmaps = all_heatmaps + hmap

        return all_heatmaps

    def detect_windows_from_heatmap(self, heatmap, threshold=1):
        '''
        Method to detect windows from heatmap
        :param heatmap: the input heatmap
        :param threshold: the minimum threshold to consider the heatmap
        :return: list of bounding boxed that meet the criteria
        '''

        result = []

        # set all values below the threshold to 0
        heatmap[heatmap <= threshold] = 0

        # use scipy's ndimage to identify labels for the different pixels
        labels = ndimg.label(heatmap)

        # for each of the labels find the non zero pixels, calculate the bounds and append to list
        for car_num in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            non_zero = (labels[0] == car_num).nonzero()

            # Identify x and y values of those pixels
            non_zero_y = np.array(non_zero[0])
            non_zero_x = np.array(non_zero[1])

            # Define a bounding box based on min/max x and y
            result.append(((np.min(non_zero_x), np.min(non_zero_y)), (np.max(non_zero_x), np.max(non_zero_y))))

        return result

    def fix_boxes(self, bboxes):
        '''
        Method to find the center of the boxes and smoothen it to avoid jitters
        :param bboxes: list of bounding boxes
        :return: the final list of bounding boxes to be used
        '''
        result = []
        old_boxes = self.boxes[:]

        # for each of the new box iterate over the old boxes and find the closest match
        # move to new centre and average the dimensions if match is found
        for box in bboxes:
            found = False

            for old_box in old_boxes:
                # check if boxes overlap
                if self.is_overlapping_box(old_box, box):
                    found = True

                    # if boxes overlap calculate new box center by getting the average
                    new_box_center = self.get_avg_center(self.get_box_center(old_box), self.get_box_center(box))

                    # get the dimensions of the boxes
                    w1, h1 = self.get_box_dim(old_box)
                    w2, h2 = self.get_box_dim(box)

                    # calculate the x and y offset
                    w = (w1 + w2) // 4
                    h = (h1 + h2) // 4

                    # calcuate the new box positions and append to the result
                    new_box = (new_box_center[0] - w, new_box_center[1] - h), (
                        new_box_center[0] + w, new_box_center[1] + h)
                    result.append(new_box)
                    break

            # if a match was not found append the box as it is to the list
            if not found:
                result.append(box)

        # if self.processed_frames % self.avg_frames == 0:
        #     self.boxes = []
        # else:
        # set the result to the recent boxes property
        self.boxes = result

        # return the final result
        return result

    def get_box_dim(self, box):
        '''
        Method to calculate the box dimensions
        :param box: the input box
        :return: the dimensions of the box
        '''

        x_left, y_left = box[0]
        x_right, y_right = box[1]

        # calculate the width and height span and return it
        return x_right - x_left, y_right - y_left

    def combine_boxes(self, bboxes):
        '''
        Recursive method to combine the boxes together if they are very near to each other
        :param bboxes: the inut boxes
        :return: the final list of boxes
        '''

        result = []
        left_over = bboxes[:]
        idx = 0
        hasCombined = False

        # repeat indefinitely until end of list is reached
        while True:
            if idx >= len(bboxes):
                break

            # get a list of all other boxes apart from the subject
            subject_box = bboxes[idx]
            compare_boxes = bboxes[idx + 1:]

            # for each of the other boxes compare if boxes are near to the subject and add
            for compare_box in compare_boxes:

                # if boxes are near add the box and attach to result
                if self.is_near(subject_box, compare_box):
                    result.append(self.add_boxes(subject_box, compare_box))

                    # mark that one combination has been made in this iteration
                    hasCombined = True

                    # remove the subject and compared boxes from the list
                    try:
                        left_over.remove(subject_box)
                        left_over.remove(compare_box)

                    except:
                        pass

            idx = idx + 1

        # append boxes to the final result
        result = result + left_over

        # if atleast one combination has been made call the function recursively to combine more boxes
        # exit criteria is when no combination has been made
        if hasCombined:
            return self.combine_boxes(result)
        else:
            return result

    def add_boxes(self, box1, box2):
        '''
        Method to add two boxes together
        :param box1: the first box
        :param box2: the second box
        :return: the final combined box
        '''

        new_box = [(), ()]

        # get the min and max from both the boxes for the new box
        new_box[0] = (min(box1[0][0], box2[0][0]), min(box1[0][1], box2[0][1]))
        new_box[1] = (max(box1[1][0], box2[1][0]), max(box1[1][1], box2[1][1]))

        # return the final box
        return new_box

    def is_overlapping_box(self, box1, box2):
        '''
        Method to check if boxes overlap
        :param box1: the first box
        :param box2: the second box
        :return: boolean indicating if boxes overlap
        '''

        # unpack the points from the boxes
        rect1_x_left, rect1_y_left = box1[0]
        rect1_x_right, rect1_y_right = box1[1]
        rect2_x_left, rect2_y_left = box2[0]
        rect2_x_right, rect2_y_right = box2[1]

        # check if the boxes do not overlap horizontally
        if rect1_x_left > rect2_x_right or rect2_x_left > rect1_x_right:
            return False

        # check if the boxes do not overlap vertically
        if rect1_y_left > rect2_y_right or rect2_y_left > rect1_y_right:
            return False

        # return overlapping as True
        return True

    def get_distance(self, a, b):
        '''
        Method to get distance between two points
        :param a: the first input point
        :param b: the second input point
        :return: the calculated distance
        '''

        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_radius(self, box):
        '''
        Method to get the radius of a box
        :param box: the input box
        :return: the radius from width and height as tuples
        '''
        x_start, y_start = box[0]
        x_end, y_end = box[1]

        return (x_end - x_start) / 2, (y_end - y_start) / 2

    def is_near(self, box1, box2):
        '''
        Method to check if boxes are near
        :param box1: the first input box
        :param box2: the second input box
        :return: boolean indicating if boxes are near
        '''

        # get the centre points of the boxes
        cx1, cy1 = self.get_box_center(box1)
        cx2, cy2 = self.get_box_center(box2)

        # get the radius of the boxes
        r1, r2 = self.get_radius(box1)
        r3, r4 = self.get_radius(box2)

        # calculate the distance from the centre
        center_distance = self.get_distance((cx1, cy1), (cx2, cy2))

        # check if the boxes are near by a threshold value based on the radius and centre distance
        if abs(center_distance - (r1 + r3)) < 30 or abs(center_distance - (r2 + r4)) < 30:
            return True

        # if abs(r1 - center_distance) < 30 or abs(r2 - center_distance) < 30:
        #     # print(abs(r1 - center_distance))
        #     # print(abs(r2 - center_distance))
        #     return True
        #
        # if abs(r3 - center_distance) < 30 or abs(r4 - center_distance) < 30:
        #     # print(abs(r3 - center_distance))
        #     # print(abs(r4 - center_distance))
        #     return True

        return False

    def get_avg_center(self, center1, center2):
        '''
        Method to get average of the center points
        :param center1: the first center point
        :param center2: the second center point
        :return:
        '''

        # return average
        return (center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2

    def get_box_center(self, box):
        '''
        Method to get the center of a box
        :param box: the input box
        :return: the center points of the box
        '''

        # unpack the points
        x_left, y_left = box[0]
        x_right, y_right = box[1]

        # calcuate centre and return
        return (x_left + x_right) // 2, (y_left + y_right) // 2

    def draw_boxes(self, img, bboxes, col=(255, 0, 0), thick=5):
        '''
        Method to draw boxes on an image
        :param img: the input image
        :param bboxes: a list of box points to draw
        :param col: the color of the box
        :param thick: the thickness
        :return: an image with boxes drawn on the input
        '''

        # use opencv's rectangle method to draw the boxes
        cpimg = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(cpimg, bbox[0], bbox[1], (255, 0, 0), thickness=thick)

        # return the output
        return cpimg

    def run(self, in_img, loaded_by_mpimg=False):
        '''
        Method to process an image and draw bounding boxes on the deteced vehicles
        :param in_img: the input image
        :param loaded_by_mpimg: boolean indicating if image was loaded by mpimg
        :return: the processed image
        '''

        # normalize the image and convert color based on the mode
        if loaded_by_mpimg:
            img = np.copy(in_img)
            norm = img.astype(np.float32) / 255
        else:
            img = np.copy(in_img)
            norm = img.astype(np.float32) / 255

        # increment the processed frames
        self.processed_frames += 1
        # cv2.imwrite('./test_images/frames/raw_' + str(self.processed_frames) + '.jpg', img)

        # define varying sliding window dimensions to search for the cars
        windows = []
        windows = self.get_all_windows(norm, x_bounds=(350, None), y_bounds=(400, 500),
                                       window_dim=(48, 48), overlap=(0.75, 0.75))
        windows += self.get_all_windows(norm, x_bounds=(350, None), y_bounds=(400, 600),
                                        window_dim=(96, 96), overlap=(0.75, 0.75))
        windows += self.get_all_windows(norm, x_bounds=(350, None), y_bounds=(400, 650),
                                        window_dim=(128, 128), overlap=(0.75, 0.75))

        # get the raw windows of prediction
        raw_windows = self.get_raw_predicted_windows(norm, windows)

        # get the heatmap and average it across recent frames
        heatmap = self.get_heatmap(norm, raw_windows)
        avg_heatmap = self.get_average_heatmap(heatmap)

        # use the heatmap to remove false positives
        heatmap_windows = self.detect_windows_from_heatmap(avg_heatmap, threshold=5)

        # combine heatmap windows to a single box if they are close to each other
        combined_heatmap_windows = self.combine_boxes(heatmap_windows)

        # get the center of the box according to previous motion and fix it to avoid too much jittering
        final_windows = self.fix_boxes(combined_heatmap_windows)

        # norm = utils.convert_color(norm, 'BGR')

        # return the processed image
        result = self.draw_boxes(img, final_windows)
        return result
