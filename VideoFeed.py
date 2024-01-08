import math

import cv2


class VideoFeed:
    def __init__(self, source, width, height, frame_buffer_size=1):
        self.width = width
        self.height = height

        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        self.cap = cap
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.frame_buffer = []
        self.frame_buffer_size = frame_buffer_size
        self.current_frame = 0

    def getDimensions(self):
        return self.width, self.height

    def getFrame(self, paused=False):
        if paused:
            out = self.frame_buffer[-1]
        else:
            has_frame, frame = self.cap.read()
            self.current_frame += 1
            if self.current_frame == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            frame = cv2.resize(frame.copy(), (self.width, self.height))
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.frame_buffer_size:
                self.frame_buffer = self.frame_buffer[1:]
            out = frame
        return out

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

    # needs either position or contour area, otherwise just returns first contour
    def getContour(self, hsv_range, position=None, contour_area=None):
        hsv_frame = cv2.cvtColor(self.frame_buffer[-1], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, hsv_range[0], hsv_range[1])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt, best_val = None, None
        if position is not None:
            for cnt in contours:
                x, y, width, height = cv2.boundingRect(cnt)
                cx, cy = x + width / 2, y + height / 2
                # could sqrt for accuracy, but its all relative so unnecessary
                dist = (position[0] - cx)**2 + (position[1] - cy)**2
                if best_cnt is None or dist < best_val:
                    best_cnt, best_val = cnt, dist
        elif contour_area is not None:
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                dist = abs(cnt_area - contour_area)
                if best_cnt is None or dist < best_val:
                    best_cnt, best_val = cnt, dist
        else:
            for cnt in contours:
                return cnt
        return best_cnt
