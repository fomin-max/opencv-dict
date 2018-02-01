import cv2
import imutils
import numpy as np
from imutils import contours


def extract_digits_and_symbols(image, charCnts, minW=1, minH=1):
    # grab the internal Python iterator for the list of character
    # contours, then  initialize the character ROI and location
    # lists, respectively
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    # keep looping over the character contours until we reach the end
    # of the list
    while True:
        try:
            # grab the next character contour from the list, compute
            # its bounding box, and initialize the ROI
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            # check to see if the width and height are sufficiently
            # large, indicating that we have found a digit
            if cW >= minW and cH >= minH:
                # extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            # otherwise, we are examining one of the special symbols
            else:
                # MICR symbols include three separate parts, so we
                # need to grab the next two parts from our iterator,
                # followed by initializing the bounding box
                # coordinates for the symbol
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
                                        -np.inf)

                # loop over the parts
                for p in parts:
                    # compute the bounding box for the part, then
                    # update our bookkeeping variables
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                # extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
                # we have reached the end of the iterator; gracefully break
                # from the loop
        except StopIteration:
            break

    # return a tuple of the ROIs and locations
    return (rois, locs)

ref = cv2.imread('rusocr.png')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = imutils.resize(ref, width=400)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
                    cv2.THRESH_OTSU)[1]
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
dict = {}

# extract the digits and symbols from the list of contours, then
# initialize a dictionary to map the character name to the ROI
refROIs = extract_digits_and_symbols(ref, refCnts, minW=1, minH=1)[0]
chars = {}
charNames = ["Ы", "А", "Н", "1", "2", "0", "Б", "Ь", "3", "В", "П", "Э", "Ю", "4", "Р", "Г", "Д", "5", "Я", "С", "6",
             "Щ", "Т", "Е", "7", "Ъ", "У", "Ё", "Ф", "Ж", "8", "Л", "9", "Х", "М", "З", "Ш", "О", "Ц", "И", "Ч", "Й", "К"]
# loop over the reference ROIs
for (name, roi) in zip(charNames, refROIs):
    # resize the ROI to a fixed size, then update the characters
    # dictionary, mapping the character name to the ROI
    roi = cv2.resize(roi, (57, 88))
    chars[name] = roi

cv2.imshow("4", chars['4'])
cv2.imshow("Ъ", chars['Ъ'])
cv2.imshow("Й", chars['Й'])
cv2.waitKey(0)