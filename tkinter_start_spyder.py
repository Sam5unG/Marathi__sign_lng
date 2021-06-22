# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:04:39 2021

@author: ****
"""

from tkinter import *
from PIL import Image, ImageFont, ImageDraw, ImageTk 
import cv2
import time
import gtts
import os
from playsound import playsound
from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.utils import plot_model

#from IPython.display import SVG, Image

import tensorflow as tf
import numpy as np

#from tkinter.ttk import *

#import prwin1
root = Tk('root')
root.geometry('640x360')
root.title('Main Screen')

label1 = Label(root, text='Marathi Sign Language Recognition', font=('bold', 25))
label1.pack()

"""
def get_img():
	

disp_img = get_img()
"""
#label_empty1

def main_window():
    root2 = Tk('root')
    root2.geometry('640x480')
    root2.title('Main Screen')

    label1 = Label(root2, text='Marathi Sign Language Detection', font=('bold', 25))
    label1.pack()


    Label(root2, text='\n\n', ).pack()

    button_2 = Button(root2, text='Letter to Sound and Sign', pady=10, command=opt2)
    button_2.pack()

    Label(root2, text='\n\n', ).pack()


    button_1 = Button(root2, text='Webcam Sign to Letter and Sound', pady=20, font=('bold', 10), command=opt1)
    button_1.pack()
    

    Label(root2, text='\n\n', ).pack()

    button_close = Button(root2, text="Close", command=root.quit)
    button_close.pack()

    root2.mainloop()
    
    
def update_label():
    label_pred['text']=time.time()
    label_pred.after(10, update_label())
    
    let_dict = {"अ":"A", "आ":"AA", "इ":"E", "ई":"EE", "उ":"U",
                "ऊ":"UU", "ए":"AE", "ऐ":"AI", "ओ":"O", "औ":"AU",
                "क":"K", "ख":"KH","ग":"G", "घ":"GH",
                "च":"C", "छ":"CHH", "ज":"J", "झ":"JH", 
                "ट":"TT", "ठ":"THH", "ड":"DD", "ढ":"DHH", "ण":"NN",
                "त":"T","थ":"TH", "द":"D", "ध":"DH", "न":"N",
                "प":"P", "फ":"F", "ब":"B", "भ":"BH","म":"M",
                "य":"Y", 'र':"R", "ल":"L", "व":"V", "श":"SH",
                "स":"S", "ह":"H", "ळ":"LL", "क्ष":"KSH", "ज्ञ":"DHNY"}
    
  
    
def opt1():
    #root.quit()
    global cap, c, ret, frame, screen1
    #screen1 = Toplevel(root)
    #screen1.geometry('64x48')
    #screen1.title('Prediction: Sign to letter')
    
    #var = model.predict(roi_img.reshape(1, 64, 64, 1))
    #label_pred = Label(screen1, text='prediction').pack()
    model = tf.keras.models.load_model(r'D:\cdac\project\Naeem_code_data\data\Marathi_hand_gesture01.h5')
    s = {'अ': 0, 'आ': 1, 'ए': 2, 'ऐ': 3, 'ब': 4, 'भ': 5, 'च': 6, 'छ': 7, 'द': 8, 'ड': 9, 'ध': 10, 'ढ': 11, 'इ': 12, 'ई': 13, 'ग': 14, 'घ': 15, 'ह': 16, 'ज': 17, 'झ': 18, 'ज्ञ': 19, 'क': 20, 'ख': 21, 'ल': 22, 'ळ': 23, 'म': 24, 'न': 25, 'ण': 26, 'ओ': 27, 'औ': 28, 'प': 29, 'फ': 30, 'र': 31, 'स': 32, 'श': 33, 'क्ष': 34, 'त': 35, 'ट': 36, 'थ': 37, 'ठ': 38, 'उ': 39, 'ऊ': 40, 'व': 41, 'य': 42}       
        
    name=[]
    for i in s:

        name.append(i)

    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        
        
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
    
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #_, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
       
        test_image = image.img_to_array(roi)

        test_image = np.expand_dims(test_image, axis = 0)
        
        result = model.predict(test_image)
         
        a=result.argmax()
        
        for i in range(43):
            #print(result[0][i])
            if(i==a):

                q=name[i]
        """
        prediction = {'A': result[0][0], 
                  'AA': result[0][1],
                  'AE': result[0][2],
                  'AI': result[0][3],
                  'AU': result[0][4],
                  'ब': result[0][5],
                  'BH': result[0][6],
                  'C': result[0][7],
                  'CHH': result[0][8],
                  'द': result[0][9],
                  'ड': result[0][10],
                  'ध': result[0][11],
                  'DHH': result[0][12],
                  'E': result[0][13],
                  'EE': result[0][14],
                  'F': result[0][15],
                  'G': result[0][16],
                  'GH': result[0][17],
                  'ह': result[0][18],
                  'ज': result[0][19],
                  'JH': result[0][20],
                  'JYA': result[0][21],
                  'क': result[0][22],
                  'KH': result[0][23],
                  'ल': result[0][24],
                  'LL': result[0][25],
                  'M': result[0][26],
                  'N': result[0][27],
                  'NN': result[0][28],
                  'O': result[0][29],
                  'P': result[0][30],
                  'र': result[0][31],
                  'S': result[0][32],
                  'SH': result[0][33],
                  'SHA': result[0][34],
                  'T': result[0][35],
                  'TH': result[0][36],
                  'THH': result[0][37],
                  'TT': result[0][38],
                  'U': result[0][39],
                  'UU': result[0][40],
                  'व': result[0][41],
                  'य': result[0][42]}
               """   
        # Sorting based on top prediction
        #max_key = max(prediction, key=prediction.get)
        #prediction = sorted(prediction.items(), key=operator.itemgetter(1))
    
        
        fontpath =r'C:\Windows\Fonts\Sanskr.ttf'
        
        #cv2.putText(frame, 'अंदाजे'+str(time.time()), (100, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        font = ImageFont.truetype(fontpath, 22) # font size 32
        img = np.zeros((500,140,3),np.uint8)
        img_pil = Image.fromarray(img) # convert each value of the array 8bit (1byte) integer type (0 to 255) in the PIL Image.
 
        draw = ImageDraw.Draw(img_pil) #the # draw instantiation
  
        position =(10, 10) # text display position
        b,g,r,a = 0,255,200,0 #B (blue) · G (green) · R (red) · A (transparency)
  
        draw.text(position, 'अंदाजे:'+q, font = font , fill = (b, g, r, a)) 
        # draw forth the text to fill Color: BGRA (RGB)
  
        img = np.array(img_pil) # PIL converted into array
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        ht, wd, c = frame.shape
        overlay =np.zeros((ht,wd,3),np.uint8)
    
        draw_ht, draw_wd, draw_c = img.shape
        for i in range(0, draw_ht):
            for j in range(0, draw_wd):
                if img[i, j][2] != 0:
                    overlay[i, j] = img[i, j]
               
        cv2.addWeighted(overlay, 1, frame, 1, 0, frame)
        
        cv2.imshow("test", roi)
        cv2.imshow('Sign to letter: Input Feed', frame)
        #update_label()
        
        c = cv2.waitKey(1)
        if c == 27:
            break
            
        #screen1.mainloop()
    #screen1.destroy()
    cap.release()
    cv2.destroyAllWindows()







    #button_home = Button(screen1, text='Main Menu', command=main_window).pack()

    #screen1.mainloop()
def speak(lettr):
    pronc = gtts.gTTS(lettr, lang='mr')
    if os.path.exists('speak.mp3'):
        os.remove('speak.mp3')
    pronc.save('speak.mp3')
    #os.system('start speak.mp3')
    
    playsound('speak.mp3')
    
def pick_ltr(dict_key):
    #pass
    global screen3, img, img_lett
    screen3 = Toplevel(screen2)
    screen3.geometry('720x640')
    screen3.title('Sign for {}'.format(dict_key))
    #img = ImageTk.PhotoImage(Image.open(r'D:\cdac\project\msl.png'))
    Button(screen3, text='उच्चार', command=lambda: speak(dict_key)).pack()
    
    let_dict = {"अ":"A", "आ":"AA", "इ":"E", "ई":"EE", "उ":"U",
                "ऊ":"UU", "ए":"AE", "ऐ":"AI", "ओ":"O", "औ":"AU",
                "क":"K", "ख":"KH","ग":"G", "घ":"GH",
                "च":"C", "छ":"CHH", "ज":"J", "झ":"JH", 
                "ट":"TT", "ठ":"THH", "ड":"DD", "ढ":"DHH", "ण":"NN",
                "त":"T","थ":"TH", "द":"D", "ध":"DH", "न":"N",
                "प":"P", "फ":"F", "ब":"B", "भ":"BH","म":"M",
                "य":"Y", 'र':"R", "ल":"L", "व":"V", "श":"SH",
                "स":"S", "ह":"H", "ळ":"LL", "क्ष":"KSH", "ज्ञ":"DHNY"}
    
    #directory = r'D:\cdac\project\DATASET_SPJ\DATASET\TEST'
    lett = let_dict[dict_key]
    #img_lett = directory+r'\{}\{}0.png'.format(lett, lett)
    img_lett = ImageTk.PhotoImage(Image.open(r'D:\cdac\project\DATASET_SPJ\DATASET\TEST\{}\{}0.png'.format(lett, lett)))
    Label(screen3, image=img_lett).pack()
    #Label(screen3, text=dict_key,).pack()
    #opt2()

    
def opt2():
    global screen2
    screen2 = Toplevel(root)
    screen2.geometry('640x480')
    screen2.title('Option: Letter to Sign')

    Button(screen2, text='अ', command=lambda: pick_ltr('अ')).grid(row=0, column=0, columnspan=1)
    Button(screen2, text='आ', command=lambda: pick_ltr('आ')).grid(row=0, column=1, columnspan=1)
    Button(screen2, text='इ', command=lambda: pick_ltr('इ')).grid(row=0, column=2, columnspan=1)
    Button(screen2, text='ई', command=lambda: pick_ltr('ई')).grid(row=0, column=3, columnspan=1)
    Button(screen2, text='उ', command=lambda: pick_ltr('उ')).grid(row=0, column=4, columnspan=1)
    Button(screen2, text='ऊ', command=lambda: pick_ltr('ऊ')).grid(row=0, column=5, columnspan=1)
    
    Button(screen2, text='ए', command=lambda: pick_ltr('ए')).grid(row=1, column=0, columnspan=1)
    Button(screen2, text='ऐ', command=lambda: pick_ltr('ऐ')).grid(row=1, column=1, columnspan=1)
    Button(screen2, text='ओ', command=lambda: pick_ltr('ओ')).grid(row=1, column=2, columnspan=1)
    Button(screen2, text='औ', command=lambda: pick_ltr('औ')).grid(row=1, column=3, columnspan=1)
    
    Button(screen2, text='क', command=lambda: pick_ltr('क')).grid(row=2, column=0, columnspan=1)
    Button(screen2, text='ख', command=lambda: pick_ltr('ख')).grid(row=2, column=1, columnspan=1)
    Button(screen2, text='ग', command=lambda: pick_ltr('ग')).grid(row=2, column=2, columnspan=1)
    Button(screen2, text='घ', command=lambda: pick_ltr('घ')).grid(row=2, column=3, columnspan=1)
    
    Button(screen2, text='च', command=lambda: pick_ltr('च')).grid(row=3, column=0, columnspan=1)
    Button(screen2, text='छ', command=lambda: pick_ltr('छ')).grid(row=3, column=1, columnspan=1)
    Button(screen2, text='ज', command=lambda: pick_ltr('ज')).grid(row=3, column=2, columnspan=1)
    Button(screen2, text='झ', command=lambda: pick_ltr('झ')).grid(row=3, column=3, columnspan=1)
    
    Button(screen2, text='ट', command=lambda: pick_ltr('ट')).grid(row=4, column=0, columnspan=1)
    Button(screen2, text='ठ', command=lambda: pick_ltr('ठ')).grid(row=4, column=1, columnspan=1)
    Button(screen2, text='ड', command=lambda: pick_ltr('ड')).grid(row=4, column=2, columnspan=1)
    Button(screen2, text='ढ', command=lambda: pick_ltr('ढ')).grid(row=4, column=3, columnspan=1)
    Button(screen2, text='ण', command=lambda: pick_ltr('ण')).grid(row=4, column=4, columnspan=1)
    
    Button(screen2, text='त', command=lambda: pick_ltr('त')).grid(row=5, column=0, columnspan=1)
    Button(screen2, text='थ', command=lambda: pick_ltr('थ')).grid(row=5, column=1, columnspan=1)
    Button(screen2, text='द', command=lambda: pick_ltr('द')).grid(row=5, column=2, columnspan=1)
    Button(screen2, text='ध', command=lambda: pick_ltr('ध')).grid(row=5, column=3, columnspan=1)
    Button(screen2, text='न', command=lambda: pick_ltr('न')).grid(row=5, column=4, columnspan=1)
    
    Button(screen2, text='प', command=lambda: pick_ltr('प')).grid(row=6, column=0, columnspan=1)
    Button(screen2, text='फ', command=lambda: pick_ltr('फ')).grid(row=6, column=1, columnspan=1)
    Button(screen2, text='ब', command=lambda: pick_ltr('ब')).grid(row=6, column=2, columnspan=1)
    Button(screen2, text='भ', command=lambda: pick_ltr('भ')).grid(row=6, column=3, columnspan=1)
    Button(screen2, text='म', command=lambda: pick_ltr('म')).grid(row=6, column=4, columnspan=1)
    
    Button(screen2, text='य', command=lambda: pick_ltr('य')).grid(row=7, column=0, columnspan=1)
    Button(screen2, text='र', command=lambda: pick_ltr('र')).grid(row=7, column=1, columnspan=1)
    Button(screen2, text='ल', command=lambda: pick_ltr('ल')).grid(row=7, column=2, columnspan=1)
    Button(screen2, text='व', command=lambda: pick_ltr('व')).grid(row=7, column=3, columnspan=1)
    Button(screen2, text='श', command=lambda: pick_ltr('श')).grid(row=7, column=4, columnspan=1)
    
    Button(screen2, text='स', command=lambda: pick_ltr('स')).grid(row=8, column=0, columnspan=1)
    Button(screen2, text='ह', command=lambda: pick_ltr('ह')).grid(row=8, column=1, columnspan=1)
    Button(screen2, text='ळ', command=lambda: pick_ltr('ळ')).grid(row=8, column=2, columnspan=1)
    Button(screen2, text='क्ष', command=lambda: pick_ltr('क्ष')).grid(row=8, column=3, columnspan=1)
    Button(screen2, text='ज्ञ', command=lambda: pick_ltr('ज्ञ')).grid(row=8, column=4, columnspan=1)
    


    #button_home = Button(screen2, text='Main Menu', command=main_window).pack()

    #screen2.mainloop()



Label(root, text='\n\n', ).pack()

button_2 = Button(root, text='Letter to Sound and Sign', pady=10, command=opt2)
button_2.pack()

Label(root, text='\n\n', ).pack()


button_1 = Button(root, text='Webcam Sign to Letter and Sound', pady=20, font=('bold', 10), command=opt1)
button_1.pack()

#label2 =Label(root)













Label(root, text='\n\n', ).pack()

#button_close = Button(root, text="Close", command=root.quit)
#button_close.pack()

root.mainloop()