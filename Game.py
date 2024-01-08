import math
import random

import cv2
import numpy as np
from cv2.typing import Scalar

from VideoFeed import VideoFeed

STATE_INIT = 0
STATE_SETUP = 1
STATE_ONGOING = 2
STATE_GAMEOVER = 3

HUE_RANGE = 10
SAT_RANGE = 120
VAL_RANGE = 120


class Game:
    def __init__(self, filename=None):
        self.window_name = "FLOAT ALONG"
        self.state = STATE_INIT

        self.capture_source = filename if not (filename is None) else 0
        self.video_feed = None

        self.paused = False
        self.should_quit = False
        self.show_mask = False

        self.ball = None
        self.ball_size = None
        self.ball_contour = None

        self.goal = None
        self.goal_size = [50, 50]

        self.track_color_bgr = None
        self.lower_color_hsv, self.upper_color_hsv = None, None

        self.width, self.height = 500, 500

    def run(self):
        if self.state == STATE_INIT:
            self.video_feed = VideoFeed(self.capture_source)
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.handleClick)

            self.width, self.height = self.video_feed.getDimensions()

            self.state = STATE_SETUP

        while not self.should_quit:
            self.handleInput()
            frame = self.video_feed.getFrame(self.paused).copy()
            self.updatePositions(frame)
            self.render(self.renderFrame(frame), self.renderOverlay(frame))

    def handleInput(self):
        input_key = cv2.waitKey(1)
        if input_key & 0xFF == 32:
            # space
            self.paused = not self.paused
        elif input_key & 0xFF == 27:
            # esc
            self.should_quit = True
        elif input_key & 0xFF == 109:
            # m
            self.show_mask = not self.show_mask
        elif input_key & 0xFF == 114:
            self.randomizeGoal()

    def updatePositions(self, frame):
        if self.track_color_bgr is not None:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, self.lower_color_hsv, self.upper_color_hsv)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area > 10:
                    x, y, width, height = cv2.boundingRect(cnt)
                    self.ball = [x, y]
                    self.ball_size = [width, height]
                    self.ball_contour = cnt

            if self.goal is not None and \
                    boxesIntersect(self.ball[0], self.ball[1], self.ball_size[0], self.ball_size[1],
                              self.goal[0], self.goal[1], self.goal_size[0], self.goal_size[1]):
                self.randomizeGoal()

    def render(self, frame, overlay):
        out = cv2.bitwise_or(frame, overlay)
        cv2.imshow(self.window_name, out)

    def renderFrame(self, frame):
        if self.track_color_bgr is not None and self.show_mask:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, self.lower_color_hsv, self.upper_color_hsv)
            return cv2.bitwise_and(frame, frame, mask=mask)
        else:
            return frame

    def renderOverlay(self, frame):
        # Get the height and width of the image
        height, width = frame.shape[:2]

        # Create a blank image of the same size
        frame_overlay = cv2.resize(np.zeros((height, width, 3), np.uint8), (width, height))

        if self.ball is not None:
            x, y = self.ball
            width, height = self.ball_size
            cv2.rectangle(frame_overlay, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.drawContours(frame_overlay, [self.ball_contour], -1, (0, 255, 0), 3)

        color = (128, 255, 120) if self.track_color_bgr is None else self.track_color_bgr.tolist()
        cv2.putText(frame_overlay, self.overlayText(), (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 4)

        if self.goal is not None:
            x1, y1 = self.goal[0], self.goal[1]
            x2, y2 = x1 + self.goal_size[0], y1 + self.goal_size[1]

            cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), (150, 255, 10), 10)

        return frame_overlay

    def overlayText(self):
        if self.state == STATE_INIT:
            return "INITIALIZING"
        elif self.state == STATE_SETUP:
            return "SETUP: click on t" \
                   "he object you want to track"
        elif self.state == STATE_ONGOING:
            return "... " + str(self.track_color_bgr)
        elif self.state == STATE_GAMEOVER:
            return "GAME OVER"

    def handleClick(self, event, x, y, flags, param):
        if flags == 1:
            # if self.state == STATE_SETUP and c is not None:
            self.track_color_bgr = self.video_feed.getBgrAtPosition(x, y)

            hsv = self.video_feed.getHsvAtPosition(x, y)
            self.lower_color_hsv = np.array([cl(hsv[0] - HUE_RANGE), cl(hsv[1] - SAT_RANGE), cl(hsv[2] - VAL_RANGE)])
            self.upper_color_hsv = np.array([fl(hsv[0] + HUE_RANGE), fl(hsv[1] + SAT_RANGE), cl(hsv[2] + VAL_RANGE)])

            self.state = STATE_ONGOING

    def randomizeGoal(self):
        self.goal = [random.randint(0, self.width), random.randint(0, self.height)]
        pass

def fl(n):
    return max(n, 0)


def cl(n):
    return min(n, 255)


def boxesIntersect(x1, y1, w1, h1, x2, y2, w2, h2):
    (a_top_x, a_top_y) = x1, y1,
    (a_bot_x, a_bot_y) = x1 + w1, y1 + h1
    (b_top_x, b_top_y) = x2, y2,
    (b_bot_x, b_bot_y) = x2 + w2, y2 + h2

    cond_1 = a_top_x < b_top_x < a_bot_x
    cond_2 = b_top_x < a_top_x < b_bot_x
    cond_3 = a_top_y < b_top_y < a_bot_y
    cond_4 = b_top_y < a_top_y < b_bot_y

    return (cond_1 or cond_2) and (cond_3 or cond_4)