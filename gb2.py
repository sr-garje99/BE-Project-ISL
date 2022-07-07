from cv2 import cv2
import numpy as np
import os


minValue = 70


#Create the directory structure
# if os.path.exists("data"):
#     #os.makedirs("data")
#     #os.makedirs("data/train")
#     #os.makedirs("data/test")
#     #os.makedirs("data/train/blank")
#os.makedirs("data/train/0")
#     #os.makedirs("data/train/1")
#     #os.makedirs("data/train/2")
#     os.makedirs("data/train/3")
#     os.makedirs("data/train/4")
#     os.makedirs("data/train/5")
# os.makedirs("data/train/6")
# os.makedirs("data/train/7")
# os.makedirs("data/train/8")
# os.makedirs("data/train/9")
# os.makedirs("data/train/A")
# os.makedirs("data/train/B")
# os.makedirs("data/train/D")
# os.makedirs("data/train/E")
# os.makedirs("data/train/F")
# os.makedirs("data/train/G")
# os.makedirs("data/train/H")
# os.makedirs("data/train/J")
# os.makedirs("data/train/K")
# os.makedirs("data/train/M")
# os.makedirs("data/train/N")
# os.makedirs("data/train/O")
# os.makedirs("data/train/P")
# os.makedirs("data/train/Q")
# os.makedirs("data/train/R")
# os.makedirs("data/train/S")
# os.makedirs("data/train/T")
# os.makedirs("data/train/U")
# os.makedirs("data/train/V")
# os.makedirs("data/train/W")
# os.makedirs("data/train/X")
# os.makedirs("data/train/Y")
# os.makedirs("data/train/Z")
#     os.makedirs("data/train/I")
#     #os.makedirs("data/test/blank")
#     os.makedirs("data/test/0")
#     #os.makedirs("data/test/1")
#     #os.makedirs("data/test/2")
#     os.makedirs("data/test/3")
#     os.makedirs("data/test/4")
#     os.makedirs("data/test/5")
#     os.makedirs("data/test/6")
#     os.makedirs("data/test/7")
#     os.makedirs("data/test/8")
#     os.makedirs("data/test/9")
#     os.makedirs("data/test/A")
#     os.makedirs("data/test/B")
#     os.makedirs("data/test/C")
#     os.makedirs("data/test/D")
#     os.makedirs("data/test/E")
#     os.makedirs("data/test/F")
#     os.makedirs("data/test/G")
#     os.makedirs("data/test/H")
#     os.makedirs("data/test/I")
    

# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read() #returns frame
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             'six': len(os.listdir(directory+"/6")),
             'A': len(os.listdir(directory+"/A")),
             'B': len(os.listdir(directory+"/B")),
             'C': len(os.listdir(directory+"/C")),
             'D': len(os.listdir(directory+"/D")),
             'E': len(os.listdir(directory+"/E")),
             'F': len(os.listdir(directory+"/F")),
             'G': len(os.listdir(directory+"/G")),
             'I': len(os.listdir(directory+"/I")),
             'H': len(os.listdir(directory+"/H")),
             'J': len(os.listdir(directory+"/J")),
             'K': len(os.listdir(directory+"/K")),
             'L': len(os.listdir(directory+"/L")),
             'M': len(os.listdir(directory+"/M")),
             'N': len(os.listdir(directory+"/N")),
             'O': len(os.listdir(directory+"/O")),
             'P': len(os.listdir(directory+"/P")),
             'Q': len(os.listdir(directory+"/Q")),
             'R': len(os.listdir(directory+"/R")),
             'S': len(os.listdir(directory+"/S")),
             'T': len(os.listdir(directory+"/T")),
             'U': len(os.listdir(directory+"/U")),
             'V': len(os.listdir(directory+"/V")),
             'W': len(os.listdir(directory+"/W")),
             'X': len(os.listdir(directory+"/X")),
             'Y': len(os.listdir(directory+"/Y")),
             'Z': len(os.listdir(directory+"/Z"))
             }
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
 #   cv2.putText(frame, "Blanks : "+str(count['blank']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    #cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "J : "+str(count['J']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "K : "+str(count['K']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
#    cv2.putText(frame, "L : "+str(count['L']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    # cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    # cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    # cv2.putText(frame, "SIX : "+str(count['six']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    # cv2.putText(frame, "SEVEN : "+str(count['seven']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    # cv2.putText(frame, "EIGHT : "+str(count['eight']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    # cv2.putText(frame, "NINE : "+str(count['nine']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "A : "+str(count['A']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "B : "+str(count['B']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
#    cv2.putText(frame, "C : "+str(count['C']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "D : "+str(count['D']), (10, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "E : "+str(count['E']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "F : "+str(count['F']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "G : "+str(count['G']), (10, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "H : "+str(count['H']), (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
#    cv2.putText(frame, "I : "+str(count['I']), (10, 480), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    
    
   # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
#    roi = cv2.resize(roi, (64, 64))
    
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    # #blur = cv2.bilateralFilter(roi,9,75,75)
    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #time.sleep(5)
    #cv2.imwrite("/home/rc/Downloads/soe/im1.jpg", roi)
    #test_image = func("/home/rc/Downloads/soe/im1.jpg")


    
    test_image = cv2.resize(test_image, (150,150))
    cv2.imshow("test", test_image)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # _,roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break


    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', test_image)


    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', test_image)    

    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'A/'+str(count['A'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'B/'+str(count['B'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'C/'+str(count['C'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'D/'+str(count['D'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'E/'+str(count['E'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'F/'+str(count['F'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'G/'+str(count['G'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'H/'+str(count['H'])+'.jpg', test_image)
    
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['I'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'J/'+str(count['J'])+'.jpg', test_image)
    
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'K/'+str(count['K'])+'.jpg', test_image)
    
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'L/'+str(count['L'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'M/'+str(count['M'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'N/'+str(count['N'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'O/'+str(count['O'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'P/'+str(count['P'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'Q/'+str(count['Q'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'R/'+str(count['R'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'S/'+str(count['S'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'T/'+str(count['T'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'U/'+str(count['U'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'V/'+str(count['V'])+'.jpg', test_image)
    
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'W/'+str(count['W'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'X/'+str(count['X'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'Y/'+str(count['Y'])+'.jpg', test_image)

    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'Z/'+str(count['Z'])+'.jpg', test_image)    
    
cap.release()
cv2.destroyAllWindows()
"""
d = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""
