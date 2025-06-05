from src.Coldect import Coldect


if __name__ == '__main__':
    # 1) load settings for column processing
    process = Coldect('GALATEA_RAW.yaml')
    # 2) load h5 file and preprocess the images, based on the settings
    process.load_file('raw_data/Données_CN_V1/GALATEA/GALATEA_C1_REF_N1.h5')
    # preprocess(True) - show the result of preprocessing on one image
    process.preprocess(False)

    # you can:
    # a) use process.detect_defects() and then process.detect_intensity() to get nice output table
    # or b) use solely process.detect_intensity - the result is the same
    # result -> pd.DataFrame with type and intensity values for each detected column on each frame
    result = process.detect_intensity()
    print(result)

    # now you can create a nice plot, using the method below:
    # this method may raise an error, when the defects are not found
    fig = process.draw_detections_on_frame('Image 0001')
    fig.show()

    # the video consisting of each frame inside h5 file can also be created:
    # normally, the video:
    # - has 2 frames per second
    # - is saved in the same folder, where the code stays under the name 'detections.mp4'
    process.create_detection_video()
