import time
import logging

import cv2
from labelling_engine import LabellingEngine

class MainEngine:
    '''
    MainEngine is a class for managing core functions.
    @params cfg: cfg
    '''
    def __init__(self, cfg):
        # Logging
        self.init_logger()

        # Variables
        self.device_number = cfg['main_engine']['device_number']
        self.padding_size = cfg['main_engine']['padding_size']
        self.window_horizontal_size = cfg['main_engine']['window_horizontal_size']
        self.window_vertical_size = cfg['main_engine']['window_vertical_size']

        # Labelling Engine
        self.lm = LabellingEngine(cfg['labelling_engine'])

    def init_logger(self):
        '''
        Initiate a logger for MainEngine.
        '''
        logger = logging.getLogger("Main.MainEngine")
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
        if (w + self.padding_size) < self.window_horizontal_size:
            w += self.padding_size
        if (h + self.padding_size) < self.window_vertical_size:
            h += self.padding_size

        return image[y:(y + h), x:(x + w)]

    def draw_bbox(self, frame, prev_time):
        '''
        draw_bbox function analyzes contours in this frame,
        detects doortag image, gets cropped image and send
        it to LabellingEngine.
            @param frame: one frame from VideoCapture (BGR image)
            @param prev_time: time when the previous draw_bbox job finished
            @return: current time
        '''
        original_img = frame
        canny = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)

        # Get all contour points
        try:
            _, coutours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        except:
            coutours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        # Analyze each contour
        for contour in coutours:
            # Get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            if self.filter_noise(x, y, w, h):
                continue

            # Draw rectangle bbox on original image
            cropped = self.crop(original_img, x, y, w, h)
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Send cropped RGB image to Labelling Engine
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)
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

    def filter_noise(self, x: int, y: int, w: int, h: int):
        '''
        Returns true if the contour is noise.
        @params x, y, w, h: size of coutour
        '''
        winH, winW = self.window_horizontal_size, self.window_vertical_size
        ratio = float(h) / float(w)

        if w > 70 or h > 40 or w < 15:
            return True
        if y > 150 or x < 200 or x > 500:
            return True
        if float(h) < float(winH) * 0.03:
            return True
        if float(w) < float(winW) * 0.05:
            return True
        if ratio < 0.45 or ratio > 0.55:
            return True
        
        return False

    def run(self):
        '''
        Main loop for capturing video and draw bbox.
        '''
        # Start and wait for predicting subprocess
        self.lm.predict_process.start()
        while not self.lm.is_ready():
            self.logger.info('waiting for LabellingEngine..')
            time.sleep(1)
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
        cv2.destroyAllWindows()
        self.logger.info('process terminated.')
