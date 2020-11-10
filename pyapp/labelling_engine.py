import logging
import datetime
from queue import Full, Empty
from multiprocessing import Process, Queue

import numpy as np
import cv2

PATH_FOR_SAVING_NOISE = './crop_noise/{}.png'
PATH_FOR_SAVING_NUM = './crop_num/{}.png'
FLAG_FOR_SAVE_IMG = False

class LabellingEngine:
    '''
    LabellingEngine is a class for managing models, queues and subprocess.
    @params cfg: cfg['labelling_engine']
    '''
    def __init__(self, cfg):
        # Logging
        self.init_logger()

        # Variables
        self.model1_cfg = cfg['model1']
        self.model2_cfg = cfg['model2']
        self.flag_for_save_img = cfg['flag_for_save_img']
        self.path_for_noise = cfg['path_for_noise']
        self.path_for_num = cfg['path_for_num']

        # Path for model files
        self.model1_name = 'checker_model.h5'
        self.model2_name = 'svhn_model.h5'

        # Set up queues
        self.image_queue = Queue(maxsize=6000)
        self.label_queue = Queue(maxsize=100)
        self.signal_queue = Queue()
        self.ready_queue = Queue()

        # Fork a subprocess
        self.predict_process = Process(target=_predict, \
            args=(self.logger, self.model1_name, self.model2_name, self.image_queue, self.label_queue, self.signal_queue, self.ready_queue))

    def init_logger(self):
        '''
        Initiate a logger for LabellingEngine.
        '''
        logger = logging.getLogger("Main.LabellingEngine")
        logger.setLevel(logging.INFO)
        self.logger = logger

    def run(self):
        '''
        Run the subprocess.
        '''
        self.predict_process.start()

    def close(self):
        '''
        Close all queues and send stop signal to subprocess.
        '''
        self.signal_queue.put_nowait('stop')
        self.image_queue.close()
        self.label_queue.close()
        self.signal_queue.close()
        self.ready_queue.close()

    def is_ready(self):
        '''
        Check that the subprocess is ready.
            @return: returns true when all models and keras package are loaded.
        '''
        try:
            msg = self.ready_queue.get_nowait()
            if msg == 'ready':
                return True
            return False
        except Empty:
            return False

    def new_tensor(self, tensor):
        '''
        Add new image tensor to image_queue.
            @param tensor: RGB image tensor (48*48*3)
        '''
        try:
            self.image_queue.put_nowait(tensor)
        except Full:
            self.logger.info('image_queue is full.')

def _preprocess_tensor(rgb_tensor):
    '''
    Convert the tensor from RGB to grayscale.
        @param rgb_tensor: RGB image tensor (48*48*3)
        @return gray: grayscale tensor (48*48*1)
        @return gray3: grayscale tensor that duplicated 3 times (48*48*3)
    '''
    rgb_tensor = np.squeeze(rgb_tensor)
    gray = np.dot(rgb_tensor[...,:3], [0.299, 0.587, 0.114])
    gray = np.reshape(gray, (48, 48, 1))
    gray /= 255.0
    gray = gray - gray.mean() # Normalize
    gray3 = np.repeat(gray, 3, axis=2) # Duplicate grayscale tensor for 3 times
    return gray, gray3

def _decode_label(_raw):
    '''
    Convert the raw output to number string
        @param _raw: raw output data
        @return: doortag number string
    '''
    ten2zero = lambda x: 0 if x == 10 else x
    raw = [str(ten2zero(it)) for it in _raw]
    if raw[0] == '0':
        return ''
    if raw[1] == '3':
        return raw[2] + raw[3] + raw[4]
    if raw[1] == '4':
        return raw[2] + raw[3] + raw[4] + '-' + raw[5]
    return ''

def _predict(logger, model1_name, model2_name, input_queue, output_queue, signal_queue, ready_queue):
    '''
    Function for subprocess that doing prediction job.
        @oaran logger: logging instance
        @param model1_name: filepath for model1
        @param model2_name: filepath for model2
        @param input_queue: queue for new image tensors
        @param output_queue: queue for output values
        @param signal_queue: signal queue for stopping the process
        @param ready_queue: queue for notifying if the process is ready
    '''
    # Lazy load keras modules
    from keras.models import load_model
    from keras.preprocessing import image

    logger.info('predict process started.')

    # Load ML models
    model1 = load_model(model1_name)
    model2 = load_model(model2_name)

    logger.info('all models are loaded.')
    ready_queue.put_nowait('ready')
    logger.info('now module is ready.')

    index = 0
    noise_count = 0
    # Main Loop
    while True:
        # Stop the main loop when stop signal detected
        try:
            sig = signal_queue.get_nowait()
            if sig == 'stop':
                logger.info('stop command detected.')
                break
        except Empty:
            pass

        # Get new image tensor
        try:
            tensor = input_queue.get_nowait()
        except Empty:
            continue

        # Log
        if noise_count % 100 == 0:
            now = datetime.datetime.now().strftime('%H:%M:%S')
            logger.info('({}) noises : {}++'.format(now, noise_count))

        # Get grayscale tensor
        tensor_1chan, tensor_3chan = _preprocess_tensor(tensor)
        tensor_1chan = np.array([tensor_1chan])
        tensor_3chan = np.array([tensor_3chan])

        # Predict that the image contains doortag via model1
        has_number = model1.predict(tensor_1chan)
        if int(has_number[0][1]) != 1:
            # Save noise image when the flag on
            if FLAG_FOR_SAVE_IMG:
                fname = PATH_FOR_SAVING_NOISE.format(index)
                index += 1
                cv2.imwrite(fname, tensor)
            noise_count += 1
            continue

        # Save doortag image when the flag on
        if FLAG_FOR_SAVE_IMG:
            fname = PATH_FOR_SAVING_NUM.format(index)
            index += 1
            cv2.imwrite(fname, tensor)

        # Predict numbers in doortag image via model2
        label_data = model2.predict(tensor_3chan)
        output = [np.argmax(it) for it in label_data]
 
        # Ignore noise image
        if (output[0] == 0) or (output[1] <= 2) or (output.count(10) >= 3):
            noise_count += 1
            continue

        # Logging
        now = datetime.datetime.now().strftime('%H:%M:%S')
        label = _decode_label(output)
        logger.info('({}) label : "{}" / raw : {}'.format(now, label, output))