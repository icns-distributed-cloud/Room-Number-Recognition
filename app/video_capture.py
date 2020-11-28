import cv2
import queue
import threading

class BufferlessVideoCapture:
    '''
    BufferlessVideoCapture is a wrapper for cv2.VideoCapture,
        which doesn't have frame buffer.
    @param name: videocapture name
    '''
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.thr = threading.Thread(target=self._reader)
        self.thr.daemon = True
        self.thr.start()

    def _reader(self):
        '''
        Main loop for thread.
        '''
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            if self.q.qsize() > 2:
                print(self.q.qsize())
            self.q.put(frame)
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def release(self):
        self.cap.release()

    def read(self):
        '''
        Read current frame.
        '''
        return True, self.q.get()
    
    def close(self):
        pass
