import random

import cv2
import numpy as np

import Util
from GameState import GameState
from VideoFeed import VideoFeed, loadImageWithOpacity, rotate_image

STATE_INIT = 0
STATE_SETUP = 1
STATE_ONGOING = 2
STATE_GAMEOVER = 3

GOAL_SCORE_DURATION = 30

CONFIG = {
    "HUE_RANGE": 10,
    "SAT_RANGE": 50,
    "VAL_RANGE": 50,
}


class Game:
    def __init__(self, filename=None):
        self.window_name = "FLOAT ALONG"
        self.gs = GameState()

        self.capture_source = filename if not (filename is None) else 0
        self.video_feed = None

        self.goal_img = loadImageWithOpacity("public/target.png")

        self.paused = False
        self.should_quit = False
        self.show_mask = False
        self.show_dev_overlay = True

        self.ball = None
        self.ball_size = None
        self.ball_contour = None

        self.isMidClick = False

        self.goal_is_scored = False
        self.goal_timer = 0
        self.goal = None
        self.goal_size = [50, 50]

        self.width, self.height = 960, 540

    def run(self):
        if self.gs.isInitializing():
            self.video_feed = VideoFeed(self.capture_source, self.width, self.height)
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.handleClickEvent)

            self.gs.setStateAwaitingSetup()

        while not self.should_quit:
            self.handleInputEvent()
            frame = self.video_feed.getFrame(self.paused).copy()
            if self.gs.isOngoing():
                self.updatePositions(frame)
                if self.goal_is_scored:
                    if self.goal_timer == 0:
                        self.goal_is_scored = False
                        self.goal_timer = -1
                        self.gs.points += 1
                        self.randomizeGoal()
                    else:
                        self.goal_timer -= 1

            self.render(self.renderFrame(frame), self.renderDevOverlay(frame), self.renderGameOverlay(frame))

    def handleInputEvent(self):
        input_key = cv2.waitKey(1)
        if input_key > 0:
            self.doInput(input_key & 0xFF)

    def handleClickEvent(self, event, x, y, flags, param):
        if not self.isMidClick:
            if flags == 1:
                if not self.isMidClick:
                    self.doClick(x, y)
                self.isMidClick = True
        else:
            if flags == 0:
                self.isMidClick = False

    def doInput(self, key):
        if key == 32:  # space
            self.paused = not self.paused
        elif key == 27:  # esc
            self.should_quit = True
        elif key == 109:  # m
            self.show_mask = not self.show_mask
        elif key == 110:  # n
            self.show_dev_overlay = not self.show_dev_overlay
        if self.gs.isSetupBall() or self.gs.isSetupField() or self.gs.isAwaitingSetup():
            if key == 122:  # z
                self.gs.setStateSetupField()
            elif key == 120:  # x
                self.gs.setStateSetupBall()
            elif key == 99:  # c
                if self.gs.isReady():
                    self.gs.setStateOngoing()
                    self.randomizeGoal()
                else:
                    self.gs.setStateAwaitingSetup()
        elif self.gs.isOngoing():
            if key == 114:  # r
                self.randomizeGoal()
        # elif key == 118:  # v
        #     pass

    def doClick(self, x, y):
        if self.gs.isSetupField():
            if self.gs.field_top_left is None:
                self.gs.field_top_left = [x, y]
            elif self.gs.field_bottom_right is None:
                self.gs.field_bottom_right = [x, y]
                self.gs.setStateAwaitingSetup()
        elif self.gs.isSetupBall():
            self.gs.ball_color_bgr = self.video_feed.getBgrAtPosition(x, y)
            hsv = self.video_feed.getHsvAtPosition(x, y)
            hsv_lower = np.array([Util.cl(hsv[0] - CONFIG["HUE_RANGE"]),
                                  Util.cl(hsv[1] - CONFIG["SAT_RANGE"]),
                                  Util.cl(hsv[2] - CONFIG["VAL_RANGE"])])
            hsv_upper = np.array([Util.fl(hsv[0] + CONFIG["HUE_RANGE"]),
                                  Util.fl(hsv[1] + CONFIG["SAT_RANGE"]),
                                  Util.cl(hsv[2] + CONFIG["VAL_RANGE"])])
            self.gs.ball_color_hsv_range = [hsv_lower, hsv_upper]
            cnt = self.video_feed.getContour(self.gs.ball_color_hsv_range, position=[x, y])
            self.gs.ball_contour_size = cv2.contourArea(cnt)

            self.gs.setStateAwaitingSetup()

    def updatePositions(self, frame):

        cnt = self.video_feed.getContour(self.gs.ball_color_hsv_range, contour_area=self.gs.ball_contour_size)
        x, y, width, height = cv2.boundingRect(cnt)
        if self.goal is not None and self.goal_is_scored == False \
                and Util.boxesIntersect(x, y,
                                        width, height,
                                        self.goal[0], self.goal[1],
                                        self.goal_size[0], self.goal_size[1]):
            # self.gs.points += 1
            self.goal_is_scored = True
            self.goal_timer = GOAL_SCORE_DURATION
            # self.randomizeGoal()

    def render(self, frame, dev_overlay, game_overlay):
        out = frame

        _, mask = cv2.threshold(cv2.cvtColor(game_overlay, cv2.COLOR_BGR2GRAY), int(10), 255, cv2.THRESH_BINARY)

        out = cv2.bitwise_or(out, out, mask=cv2.bitwise_not(mask))
        out = cv2.bitwise_or(out, game_overlay)

        if self.show_dev_overlay:
            out = cv2.bitwise_or(out, dev_overlay)

        cv2.imshow(self.window_name, out)

    def renderFrame(self, frame):
        if self.gs.ball_color_hsv_range is not None and self.show_mask:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, self.gs.ball_color_hsv_range[0], self.gs.ball_color_hsv_range[1])
            return cv2.bitwise_and(frame, frame, mask=mask)
        else:
            return frame

    def renderGameOverlay(self, frame):
        height, width = frame.shape[:2]
        frame_overlay = cv2.resize(np.zeros((height, width, 3), np.uint8), (width, height))

        if self.goal is not None:
            x1, y1 = self.goal[0], self.goal[1]
            x2, y2 = x1 + self.goal_size[0], y1 + self.goal_size[1]

            out = cv2.resize(self.goal_img, (self.goal_size[0], self.goal_size[1]))
            out = rotate_image(out, self.goal_timer*15)
            if self.goal_is_scored:
                out = out * (self.goal_timer/GOAL_SCORE_DURATION)
            frame_overlay[y1:y2, x1:x2] = out

        return frame_overlay

    def renderDevOverlay(self, frame):
        # Create a blank image of the same size as the frame
        height, width = frame.shape[:2]
        frame_overlay = cv2.resize(np.zeros((height, width, 3), np.uint8), (width, height))

        # draw field rect
        if self.gs.isFieldReady():
            cv2.rectangle(frame_overlay, self.gs.field_top_left, self.gs.field_bottom_right, (150, 255, 10), 1)

        # draw ball
        if self.gs.isBallReady():
            cnt = self.video_feed.getContour(self.gs.ball_color_hsv_range, contour_area=self.gs.ball_contour_size)
            cv2.drawContours(frame_overlay, [cnt], -1, (0, 255, 0), 3)
            x, y, width, height = cv2.boundingRect(cnt)
            # cv2.rectangle(frame_overlay, (x, y), (x + width, y + height), (0, 0, 255), 2)

        color = (128, 255, 120)  # if self.track_color_bgr is None else self.track_color_bgr.tolist()
        overlay_strings = self.overlayText()
        for i in range(len(overlay_strings)):
            overlay_string = overlay_strings[i]
            cv2.putText(frame_overlay, overlay_string, (25, 35*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 4)

        if self.goal is not None:
            x1, y1 = self.goal[0], self.goal[1]
            x2, y2 = x1 + self.goal_size[0], y1 + self.goal_size[1]

            cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), (150, 255, 10), 1)

        return frame_overlay

    def overlayText(self):
        text = []
        if self.gs.isInitializing():
            text.append("INITIALIZING")
        elif self.gs.isAwaitingSetup():
            if self.gs.field_top_left is None or self.gs.field_bottom_right is None:
                text.append(f"([Z]Field: PRESS Z TO SETUP)")
            else:
                text.append(f"([Z]Field: {self.gs.field_top_left}, {self.gs.field_bottom_right})")

            if self.gs.ball_color_bgr is None:
                text.append(f"([X]Ball: PRESS X TO SETUP)")
            else:
                text.append(f"([X]Ball: {self.gs.ball_color_bgr}, {self.gs.ball_contour_size})")

            if not self.gs.isReady():
                text.append("([C] Setup not complete. Setup field and ball)")
            else:
                text.append("([C] Setup complete. Press C to start)")
        elif self.gs.isSetupField():
            text.append("SETUP FIELD:")
            if self.gs.field_top_left is None:
                text.append("Click the top left of the field")
            else:
                text.append(f"Top Left: {self.gs.field_top_left}")
                if self.gs.field_bottom_right is None:
                    text.append("Click the bottom right of the field")
                else:
                    text.append(f"Bottom Right: {self.gs.field_bottom_right}")
        elif self.gs.isSetupBall():
            text.append("SETUP BALL:")
            text.append("Click on the color you want to track")
        elif self.gs.isOngoing():
            text.append("ONGOING:")
            text.append(f"... {self.gs.points} points")
        elif self.gs.isGameover():
            text.append("GAME OVER")
        return text

    def randomizeGoal(self):
        x = random.randint(self.gs.field_top_left[0], self.gs.field_bottom_right[0] - self.goal_size[0])
        y = random.randint(self.gs.field_top_left[1], self.gs.field_bottom_right[1] - self.goal_size[1])
        self.goal = [x, y]

