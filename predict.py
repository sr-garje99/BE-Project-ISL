from flask import Flask , render_template
import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

minValue = 70

app = Flask(__name__,template_folder='template')

@app.route("/")
def hello_world():
    return render_template('index.html')





@app.route("/gohere")
def get_prediction():
    # Loading the model
    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw.h5")
    print("Loaded model from disk")

    cap = cv2.VideoCapture(0)

    # Category dictionary
    # categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE', 6: 'SIX', 7: 'SEVEN', 8: 'EIGHT', 9: 'NINE', A: 'A', B: 'B',
    # C: 'C', D: 'D', E: 'E', F: 'F', G: 'G', H: 'H',I: 'I'}

    
    word = ""
    count = 0 
    while True:
        _, frame = cap.read()
        _, frame2 = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)

      

        frame2 = cv2.flip(frame2,1)
        
        
        # Got this from collect-data.py
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

        #cv2.imshow("Frame", frame)
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


       
        # Batch of 1
        result = loaded_model.predict(test_image.reshape(1, 150, 150, 1))
        prediction = { 'All the Best !': result[0][0],
                    'blank' : result[0][32], 
                    'Hi': result[0][1],
                    'Fine' : result[0][2],
                    'Bless you': result[0][3], 
                    'I love you': result[0][4],
                    'Excuse me' : result[0][5], 
                    'A': result[0][6],
                    'B' : result[0][7],
                    'C': result[0][8], 
                    'D': result[0][9],
                    'E' : result[0][10],
                    'F': result[0][11], 
                    'G': result[0][12],
                    'H' : result[0][13],
                    'I': result[0][14], 
                    'J': result[0][15],
                    'K' : result[0][16],
                    'L': result[0][17], 
                    'M': result[0][18],
                    'N' : result[0][19],
                    'O': result[0][20], 
                    'P': result[0][21],
                    'Q' : result[0][22],
                    'R': result[0][23], 
                    'S': result[0][24],
                    'T' : result[0][25],
                    'U' : result[0][26],
                    'V': result[0][27], 
                    'W': result[0][28],
                    'X' : result[0][29],
                    'Y' : result[0][30],
                    'Z': result[0][31]
                    
                    }
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
        # Displaying the predictions
        cv2.putText(frame, prediction[0][0], (80, 440), cv2.FONT_HERSHEY_COMPLEX , 3, (255,255,255), 1)  
        count = count + 1
        if(count>60):
            word =  word + prediction[0][0]
            print(word)
            count = 0

        #cv2.imshow("Frame", frame)
        #word = ""
        
        
        
    #100,440
        cv2.putText(frame,"word: ", (50,100) , cv2.FONT_HERSHEY_COMPLEX , 1, (255,255,255), 1)
        cv2.putText(frame,word, (150,100) , cv2.FONT_HERSHEY_COMPLEX , 1, (255,255,255), 1)
        cv2.imshow("Frame2",frame)

      


        interrupt = cv2.waitKey(2)
        if interrupt & 0xFF == 32: # esc key
            word = word + " "
        interrupt = cv2.waitKey(2)
        
        if interrupt & 0xFF == 8: #Backspace key
            word = ""
            
        
        interrupt = cv2.waitKey(2)
        if interrupt & 0xFF == 27: # esc key
            break

      

    
    cap.release()
    cv2.destroyAllWindows()
    
    
    return render_template('thankyou.html')
            
    
   

   





if __name__ == "__main__" :
    app.run(debug = True)


