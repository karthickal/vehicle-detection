from train import *
from detector import VehicleDetector
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import utils
from moviepy.editor import VideoFileClip

if __name__ == "__main__":

    # train the classifier using the image
    # clf, params, scaler = train_classifier()

    clf = get_model()
    print('model was loaded')
    params, scaler = get_train_parameters()
    print('feature parameters was loaded')

    # create a detector object with the classifier and the params
    detector = VehicleDetector(clf, params, scaler)

    # open the video clip and process each image
    source = VideoFileClip('project_video.mp4')
    output = 'project_video_out.mp4'

    # detect vehicles for each of the frame and save the output
    clip = source.fl_image(detector.run)
    clip.write_videofile(output, audio=False)

    print("{} frames were processed.".format(detector.processed_frames))



