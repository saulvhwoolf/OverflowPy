

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