import cv2


class VideoFeed:
    def __init__(self, source, frame_buffer_size=1):
        cap = cv2.VideoCapture(source)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


        self.cap = cap
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.frame_buffer = []
        self.frame_buffer_size = frame_buffer_size
        self.current_frame = 0

    def getDimensions(self):
        return self.width, self.height

    def getFrame(self, paused=False):
        if paused:
            return self.frame_buffer[-1]
        else:
            has_frame, frame = self.cap.read()
            self.current_frame += 1
            if self.current_frame == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.frame_buffer_size:
                self.frame_buffer = self.frame_buffer[1:]

            return frame

    def close(self):
        self.cap.release()

    def getBgrAtPosition(self, x, y):
        if len(self.frame_buffer) > 0:
            f = self.frame_buffer[-1]
            return f[y][x]
        return None

    def getHsvAtPosition(self, x, y):
        if len(self.frame_buffer) > 0:
            f = self.frame_buffer[-1]
            f_hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            return f_hsv[y][x]
        return None
        # cv2.destroyAllWindows()
