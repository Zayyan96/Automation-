# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY1 = {
    1: 1, 2: 4, 3: 0, 4: 3, 5: 1,
    6: 2, 7: 3, 8: 4, 9: 0, 10: 1,
    11: 2, 12: 3, 13: 4, 14: 0, 15: 1,
    16: 2, 17: 3, 18: 4, 19: 0, 20: 1,
    21: 2, 22: 3, 23: 4, 24: 0, 25: 1,
    26: 2, 27: 3, 28: 4, 29: 0, 30: 1,
    31: 2, 32: 3, 33: 4, 34: 0, 35: 1,
    36: 2, 37: 3, 38: 4, 39: 0, 40: 1,
    41: 2, 42: 3, 43: 4, 44: 0, 45: 1,
    46: 2, 47: 3, 48: 4
}

ANSWER_KEY = {
    0: ANSWER_KEY1[33],
    1: ANSWER_KEY1[17],
    2: ANSWER_KEY1[1],
    3: ANSWER_KEY1[34],
    4: ANSWER_KEY1[18],
    5: ANSWER_KEY1[2],
    6: ANSWER_KEY1[35],
    7: ANSWER_KEY1[19],
    8: ANSWER_KEY1[3],
    9: ANSWER_KEY1[36],
    10: ANSWER_KEY1[20],
    11: ANSWER_KEY1[4],
    12: ANSWER_KEY1[37],
    13: ANSWER_KEY1[21],
    14: ANSWER_KEY1[5],
    15: ANSWER_KEY1[38],
    16: ANSWER_KEY1[22],
    17: ANSWER_KEY1[6],
    18: ANSWER_KEY1[39],
    19: ANSWER_KEY1[23],
    20: ANSWER_KEY1[7],
    21: ANSWER_KEY1[40],
    22: ANSWER_KEY1[24],
    23: ANSWER_KEY1[8],
    24: ANSWER_KEY1[41],
    25: ANSWER_KEY1[25],
    26: ANSWER_KEY1[9],
    27: ANSWER_KEY1[42],
    28: ANSWER_KEY1[26],
    29: ANSWER_KEY1[10],
    30: ANSWER_KEY1[43],
    31: ANSWER_KEY1[27],
    32: ANSWER_KEY1[11],
    33: ANSWER_KEY1[44],
    34: ANSWER_KEY1[28],
    35: ANSWER_KEY1[12],
    36: ANSWER_KEY1[45],
    37: ANSWER_KEY1[29],
    38: ANSWER_KEY1[13],
    39: ANSWER_KEY1[46],
    40: ANSWER_KEY1[30],
    41: ANSWER_KEY1[14],
    42: ANSWER_KEY1[47],
    43: ANSWER_KEY1[31],
    44: ANSWER_KEY1[15],
    45: ANSWER_KEY1[48],
    46: ANSWER_KEY1[32],
    47: ANSWER_KEY1[16]
}


# load the image, convert it to grayscale, blur it
# slightly, then find edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 75, 200)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break
# Draw the contour on the original image
if docCnt is not None:
    cv2.drawContours(image, [docCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Contour1111", image)
    cv2.waitKey(0)
else:
    docCnt = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]])

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv2.imshow("warped", warped)
cv2.waitKey(0)
cv2.imshow("paper", paper)
cv2.waitKey(0)
# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(warped, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
    method="top-to-bottom")[0]
print(f"Number of questions detected: {len(questionCnts) // 5}")

correct = 0
max_total = 0  # Initialize max_total to store the maximum non-zero pixels
# each question has 5 possible answers, to loop over the
# question in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sort the contours for the current question from
    # left to right, then initialize the index of the
    # bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]

    # Display each contour (MCQ choice) after sorting
    for c in cnts:
        # Draw the contour on a copy of the paper image
        paper_copy = paper.copy()
        cv2.drawContours(paper_copy, [c], -1, (0, 255, 0), 25)
        cv2.imshow("Sorted Contour", paper_copy)
        cv2.waitKey(0)

    bubbled = None

    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        
        # construct a mask that reveals only the current
        # "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # apply the mask to the thresholded image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        print(f"Question {q+1}, Choice {chr(65+j)}: Total non-zero pixels = {total}")

        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
            print(f"New bubbled for Question {q+1}: Total = {total}, Index = {j}")
            max_total = max(max_total, total)
    # Update max_total if the current total is greater
    

    # initialize the contour color and the index of the
    # correct answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    print(f"Question {q+1}: k={k}, bubbled={bubbled[1]}, max_total = {max_total}")  # Print k and bubbled[1] before the if statement

    # check to see if the bubbled answer is correct
    if max_total <= 1000:  # If no choice is filled in
        cv2.drawContours(paper, [cnts[k]], -1, (0, 0, 255), 30)  # Mark correct answer in red
        print(f"Question {q+1}: No answer selected. Correct answer: {chr(65+k)}")
    elif k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
        print(f"Question {q+1}: Correct answer selected: {chr(65+k)}")  # chr(65) is 'A', chr(66) is 'B', and so on
    else:
        print(f"Question {q+1}: Incorrect answer selected. Correct answer: {chr(65+k)}, Selected answer: {chr(65+bubbled[1])}")
    max_total = 0 
    # draw the outline of the correct answer on the test
    cv2.drawContours(paper, [cnts[k]], -1, color, 30)

# grab the test taker
score = (correct / 48) * 100
print("[INFO] score: {:.2f}%".format(score))
print(f"Maximum non-zero pixels for a question: {max_total}")
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
