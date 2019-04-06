'''
Author: Manuel Vasquez
'''

from matplotlib import pyplot as plt
import numpy as np
import cv2


def optical_flow(before, after):
    feature_params = dict(maxCorners=100, qualityLevel=.3,
                          minDistance=7, blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))
    initial_frame = cv2.imread(before)
    initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    points_0 = cv2.goodFeaturesToTrack(
        initial_gray, mask=None, **feature_params)
    mask = np.zeros_like(initial_frame)

    next_frame = cv2.imread(after)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    points_1, status, _ = cv2.calcOpticalFlowPyrLK(
        initial_gray, next_gray, points_0, None, **lk_params)

    good_next = points_1[status == 1]
    good_initial = points_0[status == 1]

    for i, (i_next, i_initial) in enumerate(zip(good_next, good_initial)):
        line_start = tuple(i_next.ravel())
        line_end = tuple(i_initial.ravel())
        mask = cv2.line(mask, line_start, line_end, color[i].tolist(), 2)
        next_frame = cv2.circle(next_frame, line_start,
                                4, color[i].tolist(), -1)
    img = cv2.add(next_frame, mask)

    return img


def main():
    '''
    main
    '''

    img = optical_flow('basketball1.png', 'basketball2.png')
    cv2.imshow('lucas kanade', img)

    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()
