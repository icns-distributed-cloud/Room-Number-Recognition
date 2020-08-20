import time
import logging

import cv2
from labeling_module import LabelingModule

GUI_WINDOW_SIZE = (640, 480)

class MainModule:
    '''
    MainModule is a class for managing core functions.
    '''
    def __init__(self, device_number, padding_size):
        # Logging
        self.init_logger()

        # Labeling Module
        self.lm = LabelingModule()

        # Variables
        self.device_number = device_number
        self.padding_size = padding_size

    def init_logger(self):
        '''
        Initiate a logger for MainModule.
        '''
        logger = logging.getLogger("Main.MainModule")
        logger.setLevel(logging.INFO)
        self.logger = logger

    def crop(self, image, x, y, w, h):
        '''
        Return cropped image with padding.
            @param image: image 2D array
            @param x: coordinate for x-axis
            @param y: coordinate for y-axis
            @param w: width
            @param h: height
            @return: cropped image array
        '''
        if x > self.padding_size:
            x -= self.padding_size
        if y > self.padding_size:
            y -= self.padding_size
        if (w + self.padding_size) < GUI_WINDOW_SIZE[0]:
            w += self.padding_size
        if (h + self.padding_size) < GUI_WINDOW_SIZE[1]:
            h += self.padding_size

        return image[y:(y + h), x:(x + w)]

    def draw_bbox(self, frame, prev_time):
        '''
        draw_bbox function analyzes contours in this frame,
        detects doortag image, gets cropped image and send
        it to LabelingModule.
            @param frame: one frame from VideoCapture (BGR image)
            @param prev_time: time when the previous draw_bbox job finished
            @return: current time
        '''
        original_img = frame
        canny = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)

        # Get all contour points
        try:
            _, coutours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        except:
            coutours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Analyze each contour
        for contour in coutours:
            # Get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Get doortag image
            if (w > 70) or (h > 45) or (w < 15):
                continue
            if (h / w > 0.7) or (w / h > 1.8):
                continue
            if (h > 40) or (w > 70):
                continue
            if (y > 150) or (x > 500) or (x < 200):
                continue

            # Draw rectangle bbox on original image
            cropped = self.crop(original_img, x, y, w, h)
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # Send cropped RGB image to Labeling Module
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = cv2.resize(cropped, (48, 48))
            self.lm.new_tensor(cropped)
        
        # Calculate FPS and show FPS string on frame
        current_time = time.time()
        sec = current_time - prev_time
        try:
            fps = 1/sec
        except ZeroDivisionError:
            fps = 0
        fps_string = 'FPS : {}'.format(fps)
        cv2.putText(original_img, fps_string, (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, \
            0.8, (0, 255, 0), 1)
        cv2.imshow('result', original_img)

        return current_time

    def run(self):
        '''
        Main loop for capturing video and draw bbox.
        '''
        # Start and wait for predicting subprocess
        self.lm.predict_process.start()
        while not self.lm.is_ready():
            self.logger.info('waiting for LabelingModule..')
            time.sleep(3)
        self.logger.info('now accessing to camera device..')

        # Main loop
        cap = cv2.VideoCapture(self.device_number)
        while cap.isOpened():
            try:
                # Get frame
                ret, inp = cap.read()
                if ret:
                    prev_time = self.draw_bbox(inp, prev_time)

                # Wait for quit signal
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info('terminating process..')
                    break
            except KeyboardInterrupt:
                self.logger.info('terminating process.. (occured by KeyboardInterrupt)')
                break

        # Release device and wait for closing the subprocess
        cap.release()
        self.lm.close()
        self.lm.predict_process.join()
        self.logger.info('process terminated.')
