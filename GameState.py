
S_INIT = 0
S_AWAITING_SETUP = 1
S_SETUP_FIELD = 2
S_SETUP_BALL = 3
S_ONGOING = 4
S_GAMEOVER = 5

# SETUP_INIT = 0
# SETUP_COLOR_TRACKER = 1
# SETUP_FIELD_AREA = 2


class GameState:

    def __init__(self):
        self.state = S_INIT
        self.ready = False

        self.points = 0

        self.field_top_left = None
        self.field_bottom_right = None

        self.goal_top_left = None
        self.goal_bottom_right = None

        self.ball_top_left = None
        self.ball_bottom_right = None
        self.ball_color_bgr = None
        self.ball_color_hsv_range = None
        self.ball_contour_size = None

    def isInitializing(self):
        return self.state == S_INIT

    def isAwaitingSetup(self):
        return self.state == S_AWAITING_SETUP

    def isSetupField(self):
        return self.state == S_SETUP_FIELD

    def isSetupBall(self):
        return self.state == S_SETUP_BALL

    def isOngoing(self):
        return self.state == S_ONGOING

    def isGameover(self):
        return self.state == S_GAMEOVER

    def setStateAwaitingSetup(self):
        self.state = S_AWAITING_SETUP
        self.ready = False
        if self.isBallReady() and self.isFieldReady():
            self.ready = True

    def setStateSetupField(self):
        self.state = S_SETUP_FIELD

        self.field_top_left = None
        self.field_bottom_right = None

    def isFieldReady(self):
        return self.field_top_left is not None and self.field_bottom_right is not None

    def setStateSetupBall(self):
        self.state = S_SETUP_BALL

        self.ball_color_bgr = None
        self.ball_color_hsv_range = None

        self.ball_top_left = None
        self.ball_bottom_right = None

        self.ball_contour_size = None

    def isBallReady(self):
        return self.ball_color_bgr is not None and self.ball_color_hsv_range is not None

    def setStateOngoing(self):
        self.state = S_ONGOING
        self.points = 0

    def isReady(self):
        return self.isBallReady() and self.isFieldReady()


