import cv2

import numpy as np
import math


cam = cv2.VideoCapture(0)
while(cam.isOpened()):
    # reading image from camera
    ret, image = cam.read()
    image = cv2.flip(image,1)

    # marking the region of interest
    cv2.rectangle(image, (300,300), (100,100), (0,255,0),0)
    cropped_image = image[100:300, 100:300]

    # converting BGR image to grayscale image
    gray_scaled = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # blurring the image for smoothening and reducing the noise
    blurred_image = cv2.GaussianBlur(gray_scaled, (35, 35), 0)

    # thresholding the image 
    _, threshold_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow('Thresholded', threshold_image)

    contours, h = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # finding contour with maximum area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # finding convex hull
    convex_hull = cv2.convexHull(cnt)
    
    # finding area of hand and area of convex hull
    area_of_cnt = cv2.contourArea(cnt)
    area_of_hull = cv2.contourArea(convex_hull)
      
    # finding percentage of area not covered by hand in convex hull
    area_ratio=((area_of_hull-area_of_cnt)/area_of_hull)*100

    # drawing contours
    drawing = np.zeros(cropped_image.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [convex_hull], 0,(0, 0, 255), 0)

    # finding convex hull
    convex_hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, convex_hull)
    
    no_of_defects = 0
    # counting no. of defects formed between fingers
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            start_point = tuple(cnt[s][0])
            end_point = tuple(cnt[e][0])
            far_point = tuple(cnt[f][0])

            # finding sides of triangle
            a = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            b = math.sqrt((far_point[0] - start_point[0])**2 + (far_point[1] - start_point[1])**2)
            c = math.sqrt((end_point[0] - far_point[0])**2 + (end_point[1] - far_point[1])**2)
            # finding semi-perimeter of triangle
            s = (a+b+c)/2
            # finding area of triangle
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between far point and convex hull
            dist=(2*ar)/a

            # using cosine rule to find the angle between fingers
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            # ignoring angles greater than 90 degree
            # because two fingers always form acute angle between them
            if angle <= 90 and dist > 30:
                no_of_defects += 1
                cv2.circle(cropped_image, far_point, 3, [0,0,255], -1)

            cv2.line(cropped_image,start_point, end_point, [0,255,0], 2)

        no_of_fingers = no_of_defects + 1

        # displaying results based on calculated values of no_of_fingers and area_ratio
        if no_of_fingers == 1:
            if area_ratio<12:
                cv2.putText(image,"0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            elif area_ratio<25:
                cv2.putText(image,"All the best!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            else:
                cv2.putText(image,"1 finger", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        elif no_of_fingers == 2:
            cv2.putText(image, "2 fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        elif no_of_fingers == 3:
            cv2.putText(image,"3 fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        elif no_of_fingers == 4:
            cv2.putText(image,"4 fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        elif no_of_fingers == 5:
            cv2.putText(image,"Entire hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        else:
            cv2.putText(image,"Try again", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        cv2.imshow('Gesture', image)
        cv2.imshow('Contours', np.hstack((drawing, cropped_image)))

        k = cv2.waitKey(1)
        if k == 27:
            break

cv2.destroyAllWindows()
cam.release() 