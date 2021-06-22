import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/A")
    os.makedirs("data/train/Aa")
    os.makedirs("data/train/E")
    os.makedirs("data/train/Ei")
    os.makedirs("data/train/U")
    os.makedirs("data/train/Uu")
    os.makedirs("data/train/Ae")
    os.makedirs("data/train/Aei")
    os.makedirs("data/train/O")
    os.makedirs("data/train/Ou")
    os.makedirs("data/train/K")
    os.makedirs("data/train/Kh")
    os.makedirs("data/train/G")
    os.makedirs("data/train/Gh")
    os.makedirs("data/train/Ch")
    os.makedirs("data/train/Chh")
    os.makedirs("data/train/J")
    os.makedirs("data/train/Jh")
    os.makedirs("data/train/Ta")
    os.makedirs("data/train/Tha")
    os.makedirs("data/train/Da")
    os.makedirs("data/train/Dha")
    os.makedirs("data/train/Na")
    os.makedirs("data/train/T")
    os.makedirs("data/train/Th")
    os.makedirs("data/train/D")
    os.makedirs("data/train/Dh")
    os.makedirs("data/train/N")
    os.makedirs("data/train/P")
    os.makedirs("data/train/Ph")
    os.makedirs("data/train/B")
    os.makedirs("data/train/Bh")
    os.makedirs("data/train/M")
    os.makedirs("data/train/Y")
    os.makedirs("data/train/R")
    os.makedirs("data/train/L")
    os.makedirs("data/train/V")
    os.makedirs("data/train/Sh")
    os.makedirs("data/train/S")
    os.makedirs("data/train/H")
    os.makedirs("data/train/La")
    os.makedirs("data/train/Sha")
    os.makedirs("data/train/Jya")
    os.makedirs("data/test/A")
    os.makedirs("data/test/Aa")
    os.makedirs("data/test/E")
    os.makedirs("data/test/Ei")
    os.makedirs("data/test/U")
    os.makedirs("data/test/Uu")
    os.makedirs("data/test/Ae")
    os.makedirs("data/test/Aei")
    os.makedirs("data/test/O")
    os.makedirs("data/test/Ou")
    os.makedirs("data/test/K")
    os.makedirs("data/test/Kh")
    os.makedirs("data/test/G")
    os.makedirs("data/test/Gh")
    os.makedirs("data/test/Ch")
    os.makedirs("data/test/Chh")
    os.makedirs("data/test/J")
    os.makedirs("data/test/Jh")
    os.makedirs("data/test/Ta")
    os.makedirs("data/test/Tha")
    os.makedirs("data/test/Da")
    os.makedirs("data/test/Dha")
    os.makedirs("data/test/Na")
    os.makedirs("data/test/T")
    os.makedirs("data/test/Th")
    os.makedirs("data/test/D")
    os.makedirs("data/test/Dh")
    os.makedirs("data/test/N")
    os.makedirs("data/test/P")
    os.makedirs("data/test/Ph")
    os.makedirs("data/test/B")
    os.makedirs("data/test/Bh")
    os.makedirs("data/test/M")
    os.makedirs("data/test/Y")
    os.makedirs("data/test/R")
    os.makedirs("data/test/L")
    os.makedirs("data/test/V")
    os.makedirs("data/test/Sh")
    os.makedirs("data/test/S")
    os.makedirs("data/test/H")
    os.makedirs("data/test/La")
    os.makedirs("data/test/Sha")
    os.makedirs("data/test/Jya")
    
    

# Choosing train or test folder
mode = 'test'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'A': len(os.listdir(directory+"/A")),
             'Aa': len(os.listdir(directory+"/Aa")),
             'E': len(os.listdir(directory+"/E")),
             'Ei': len(os.listdir(directory+"/Ei")),
             'U': len(os.listdir(directory+"/U")),
             'Uu': len(os.listdir(directory+"/Uu")),
             'Ae': len(os.listdir(directory+"/Ae")),
             'Aei': len(os.listdir(directory+"/Aei")),
             'O': len(os.listdir(directory+"/O")),
             'Ou': len(os.listdir(directory+"/Ou")),
             'K': len(os.listdir(directory+"/K")),
             'Kh': len(os.listdir(directory+"/Kh")),
             'G': len(os.listdir(directory+"/G")),
             'Gh': len(os.listdir(directory+"/Gh")),
             'Ch': len(os.listdir(directory+"/Ch")),
             'Chh': len(os.listdir(directory+"/Chh")),
             'J': len(os.listdir(directory+"/J")),
             'Jh': len(os.listdir(directory+"/Jh")),
             'Ta': len(os.listdir(directory+"/Ta")),
             'Tha': len(os.listdir(directory+"/Tha")),
             'Da': len(os.listdir(directory+"/Da")),
             'Dha': len(os.listdir(directory+"/Dha")),
             'Na': len(os.listdir(directory+"/Na")),
             'T': len(os.listdir(directory+"/T")),
             'Th': len(os.listdir(directory+"/Th")),
             'D': len(os.listdir(directory+"/D")),
             'Dh': len(os.listdir(directory+"/Dh")),
             'N': len(os.listdir(directory+"/N")),
             'P': len(os.listdir(directory+"/P")),
             'Ph': len(os.listdir(directory+"/Ph")),
             'B': len(os.listdir(directory+"/B")),
             'Bh': len(os.listdir(directory+"/Bh")),
             'M': len(os.listdir(directory+"/M")),
             'Y': len(os.listdir(directory+"/Y")),
             'R': len(os.listdir(directory+"/R")),
             'L': len(os.listdir(directory+"/L")),
             'V': len(os.listdir(directory+"/V")),
             'Sh': len(os.listdir(directory+"/Sh")),
             'S': len(os.listdir(directory+"/S")),
             'H': len(os.listdir(directory+"/H")),
             'La': len(os.listdir(directory+"/La")),
             'Sha': len(os.listdir(directory+"/Sha")),
             'Jya': len(os.listdir(directory+"/Jya"))}
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " A : "+str(count['A']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Aa : "+str(count['Aa']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " E : "+str(count['E']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Ei : "+str(count['Ei']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " U : "+str(count['U']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Uu : "+str(count['Uu']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Ae : "+str(count['Ae']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Aei : "+str(count['Aei']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " O : "+str(count['O']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Ou : "+str(count['Ou']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " K : "+str(count['K']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Kh : "+str(count['Kh']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " G : "+str(count['G']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Gh : "+str(count['Gh']), (10, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Ch : "+str(count['Ch']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Chh : "+str(count['Chh']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " J : "+str(count['J']), (10, 430), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Jh : "+str(count['Jh']), (10, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Ta : "+str(count['Ta']), (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Tha : "+str(count['Tha']), (100, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Da : "+str(count['Da']), (100, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Dha : "+str(count['Dha']), (100, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Na : "+str(count['Na']), (100, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " T : "+str(count['T']), (100, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Th : "+str(count['Th']), (100, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " D : "+str(count['D']), (100, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Dh : "+str(count['Dh']), (100, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " N : "+str(count['N']), (100, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " P : "+str(count['P']), (100, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Ph : "+str(count['Ph']), (100, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " B : "+str(count['B']), (100, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Bh : "+str(count['Bh']), (100, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " M : "+str(count['M']), (100, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Y : "+str(count['Y']), (100, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " R : "+str(count['R']), (100, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " L : "+str(count['L']), (100, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " V : "+str(count['V']), (100, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Sh : "+str(count['Sh']), (200, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " S : "+str(count['S']), (200, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " H : "+str(count['H']), (200, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " La : "+str(count['La']), (200, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Sha : "+str(count['Sha']), (200, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, " Jya : "+str(count['Jya']), (200, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
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
    roi = cv2.resize(roi, (64, 64)) 
 
    cv2.imshow("Frame", frame)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #_, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'A/'+str(count['A'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'Aa/'+str(count['Aa'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'E/'+str(count['E'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'Ei/'+str(count['Ei'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'U/'+str(count['U'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'Uu/'+str(count['Uu'])+'.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'Ae/'+str(count['Ae'])+'.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'Aei/'+str(count['Aei'])+'.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'O/'+str(count['O'])+'.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'Ou/'+str(count['Ou'])+'.jpg', roi)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'K/'+str(count['K'])+'.jpg', roi)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'Kh/'+str(count['Kh'])+'.jpg', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'G/'+str(count['G'])+'.jpg', roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'Gh/'+str(count['Gh'])+'.jpg', roi)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'Ch/'+str(count['Ch'])+'.jpg', roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'Chh/'+str(count['Chh'])+'.jpg', roi)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'J/'+str(count['J'])+'.jpg', roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'Jh/'+str(count['Jh'])+'.jpg', roi)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'Ta/'+str(count['Ta'])+'.jpg', roi)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'Tha/'+str(count['Tha'])+'.jpg', roi)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'Da/'+str(count['Da'])+'.jpg', roi)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'Dha/'+str(count['Dha'])+'.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'Na/'+str(count['Na'])+'.jpg', roi)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'T/'+str(count['T'])+'.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'Th/'+str(count['Th'])+'.jpg', roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'D/'+str(count['D'])+'.jpg', roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'Dh/'+str(count['Dh'])+'.jpg', roi)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'N/'+str(count['N'])+'.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'P/'+str(count['P'])+'.jpg', roi)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'Ph/'+str(count['Ph'])+'.jpg', roi)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'B/'+str(count['B'])+'.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'Bh/'+str(count['Bh'])+'.jpg', roi)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'M/'+str(count['M'])+'.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'Y/'+str(count['Y'])+'.jpg', roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'R/'+str(count['R'])+'.jpg', roi)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'L/'+str(count['L'])+'.jpg', roi)
    if interrupt & 0xFF == ord(','):
        cv2.imwrite(directory+'V/'+str(count['V'])+'.jpg', roi)
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(directory+'Sh/'+str(count['Sh'])+'.jpg', roi)
    if interrupt & 0xFF == ord('/'):
        cv2.imwrite(directory+'S/'+str(count['S'])+'.jpg', roi)
    if interrupt & 0xFF == ord('['):
        cv2.imwrite(directory+'H/'+str(count['H'])+'.jpg', roi)
    if interrupt & 0xFF == ord(']'):
        cv2.imwrite(directory+'La/'+str(count['La'])+'.jpg', roi)
    if interrupt & 0xFF == ord(';'):
        cv2.imwrite(directory+'Sha/'+str(count['Sha'])+'.jpg', roi)
    if interrupt & 0xFF == ord('-'):
        cv2.imwrite(directory+'Jya/'+str(count['Jya'])+'.jpg', roi)
    
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
