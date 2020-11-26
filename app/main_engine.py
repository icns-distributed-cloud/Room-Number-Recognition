import time
import logging

import cv2
from labelling_engine import LabellingEngine
# from serial_engine import SerialEngine
from mqtt_engine import MQTTEngine

class MainEngine:
    '''
    MainEngine is a class for managing core functions.
    @param cfg: cfg
    '''
    def __init__(self, cfg):
        # Logging
        self.init_logger()

        # Variables
        self.device_number = cfg['main_engine']['device_number']
        self.padding_size = cfg['main_engine']['padding_size']
        self.window_horizontal_size = cfg['main_engine']['window_horizontal_size']
        self.window_vertical_size = cfg['main_engine']['window_vertical_size']
        self.fps_queue = []
        self.fps_queue_cap = cfg['main_engine']['fps_queue_capacity']
        self.most_frequent_label = ''
        self.noise_counter = 0
        self.noise_counter_threshold = cfg['main_engine']['noise_counter_threshold']
        self.show_on_gui = cfg['main_engine']['show_on_gui']

        # Labelling Engine
        self.le = LabellingEngine(cfg['labelling_engine'])
        # self.se = SerialEngine('/dev/ttyS0', 9600)
        self.mqtt = MQTTEngine(cfg['mqtt_engine'])
        self.mqtt.connect()

    def init_logger(self):
        '''
        Initiate a logger for MainEngine.
        '''
        logger = logging.getLogger("Main.MainEngine")
        logger.setLevel(logging.INFO)
        self.logger = logger

    def crop(self, image, x, y, w, h, padding):
        '''
        Return cropped image with padding.
            @param image: image 2D array
            @param x: coordinate for x-axis
            @param y: coordinate for y-axis
            @param w: width
            @param h: height
            @param padding: padding size
            @return: cropped image array
        '''
        copied = image.copy()
        if x > padding:
            x -= padding
        if y > padding:
            y -= padding
        if (w + padding * 2) < self.window_horizontal_size:
            w += padding * 2
        else:
            w = self.window_horizontal_size
        if (h + padding * 2) < self.window_vertical_size:
            h += padding * 2
        else:
            h = self.window_vertical_size

        return copied[y:(y + h), x:(x + w)]

    def draw_bbox(self, frame, prev_time):
        '''
        draw_bbox function analyzes contours in this frame,
        detects doortag image, gets cropped image and send
        it to LabellingEngine.
            @param frame: one frame from VideoCapture (BGR image)
            @param prev_time: time when the previous draw_bbox job finished
            @return: current time
            @return: True if the frame contains number
        '''
        original_img = frame
        canny = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)

        # Get all contour points
        try:
            _, coutours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        except:
            coutours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        # Analyze each contour
        found_number = False
        for contour in coutours:
            # Get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            if self.filter_noise(x, y, w, h):
                continue

            # Send cropped RGB image to Labelling Engine
            cropped = self.crop(original_img, x, y, w, h, self.padding_size)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)
            bigcrop = self.crop(original_img, x, y, w, h, 100)
            bigcrop = cv2.cvtColor(bigcrop, cv2.COLOR_BGR2RGB)
            label, self.most_frequent_label, ok = self.le.predict(cropped, bigcrop)
            label_string = label
            cv2.putText(original_img, label_string, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, \
                0.9, (0, 0, 255), 1)

            # Draw rectangle bbox on original image
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if not ok:
                continue

            found_number = True
            print('label: {} / most: {}'.format(label, self.most_frequent_label))
        
        # Calculate FPS and show FPS string on frame
        current_time = time.time()
        sec = current_time - prev_time
        fps = self.calc_fps(sec)
        fps_string = 'FPS : {}'.format(fps)
        cv2.putText(original_img, fps_string, (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, \
            0.8, (0, 255, 0), 1)

        # Show the most frequent label
        freq_string = 'Most Frequent Label : {}'.format(self.most_frequent_label)
        cv2.putText(original_img, freq_string, (5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, \
            0.8, (0, 255, 0), 1)

        if self.show_on_gui:
            cv2.imshow('Room Number Recognition', original_img)

        return current_time, found_number
    
    def clear_most_frequent_label(self):
        '''
        Clear noise counter and the most frequent label.
        '''
        self.noise_counter = 0
        self.most_frequent_label = ''
        self.le.clear_most_frequent_label()

    def filter_noise(self, x: int, y: int, w: int, h: int):
        '''
        Returns true if the contour is noise.
        @param x, y, w, h: size of coutour
        '''
        winH, winW = self.window_horizontal_size, self.window_vertical_size
        ratio = float(h) / float(w)

        if w > 70 or h > 40 or w < 15:
            return True
        # if y > 150 or x < 200 or x > 500:
        #     return True
        if float(h) < float(winH) * 0.03:
            return True
        if float(w) < float(winW) * 0.05:
            return True
        if ratio < 0.45 or ratio > 0.55:
            return True
        
        return False

    def calc_fps(self, now_fps=None):
        '''
        Returns average FPS.
        @param now_fps: elapsed time
        @return: average FPS
        '''
        func = lambda q: round(len(q) / sum(q), 1)
        if now_fps is None:
            if len(self.fps_queue) == 0:
                return 1
            return func(self.fps_queue)
        
        self.fps_queue.append(now_fps)
        if len(self.fps_queue) > self.fps_queue_cap:
            self.fps_queue = self.fps_queue[1:]
        return func(self.fps_queue)

    def run(self):
        '''
        Main loop for capturing video and draw bbox.
        '''
        # Start and wait for predicting subprocess
        self.logger.info('now accessing to camera device..')

        # cv2 window
        if self.show_on_gui:
            cv2.namedWindow('Room Number Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Room Number Recognition', self.window_horizontal_size, self.window_vertical_size)

        # Main loop
        cap = cv2.VideoCapture(self.device_number)
        prev_time = 1
        while cap.isOpened():
            try:
                # Get frame
                ret, inp = cap.read()
                if ret:
                    prev_time, found_number = self.draw_bbox(inp, prev_time)
                    if not found_number:
                        self.noise_counter += 1
                        if self.noise_counter > self.noise_counter_threshold:
                            self.clear_most_frequent_label()
                    else:
                        self.mqtt.publish({
                            'label': str(self.most_frequent_label)
                        })

                # Wait for quit signal
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info('terminating process..')
                    break
            except KeyboardInterrupt:
                self.logger.info('terminating process.. (occured by KeyboardInterrupt)')
                break

        # Release device and wait for closing the subprocess
        cap.release()
        self.logger.info('released camera.')
        self.le.close()
        self.logger.info('closed LabellingEngine.')
        cv2.destroyAllWindows()
        self.logger.info('closed all windows.')
