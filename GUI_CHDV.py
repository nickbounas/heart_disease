from os import path
import tkinter as tk
from tkinter.constants import ANCHOR, BOTH, COMMAND, LEFT, RIGHT, SCROLL, X, Y
from tracemalloc import stop
from typing import BinaryIO, Text
import urllib
from numpy import common_type
from PIL import Image, ImageTk
from numpy.lib.polynomial import roots
import urllib.request
import numpy as np
import pickle
flag = 0

def ende():
    fenster.destroy()

## Function submit button
def submit():
    
    ## The result window
    second = tk.Tk()

    ## Input variables from the GUI
    try:
        VModel = li_Model.get(li_Model.curselection()) 
        if VModel == 'Neural Network':
            flag = 0
        else: 
            flag = 1

        VSex = li_SEX.get(li_SEX.curselection())
        if VSex == 'male':
            VSex = 1 # male
            VSex = int(VSex)
        elif VSex == 'female':
            VSex = 0 # female
            VSex = int(VSex)
    
        VAge = int(txt_AGE.get())

        VCSmoker = li_CSmoker.get(li_CSmoker.curselection())
        if VCSmoker == 'yes':
            VCSmoker = 1 # yes
        else:
            VCSmoker = 0 # no
        VCSmoker = int(VCSmoker)


        VDiabetes = li_Diabetes.get(li_Diabetes.curselection())
        if VDiabetes == 'yes':
            VDiabetes = 1 # yes
        else:
            VDiabetes = 0 # no
        VDiabetes = int(VDiabetes)

        VTotChol = int(txt_TotChol.get())
        VSysBP = int(txt_SysBP.get())
        VDiaBP = int(txt_DiaBP.get())
        # if flag == 0:
        #     VCigsPerDay = int(txt_CigsPerDay.get())
        #     VBPMeds = int(txt_BPMeds.get())
        #     VBMI = int(txt_BMI.get())
        #     VHeartRate = int(txt_HeartRate.get())
        #     VGlucose = int(txt_Glucose.get())

        #     if VPrevalentStroke == 'yes':
        #         VPrevalentStroke = 1 # yes
        #     else:
        #         VPrevalentStroke = 0 # no
        #     VPrevalentStroke = int(VPrevalentStroke)

        #     if VPrevalentHyp == 'yes':
        #         VPrevalentHyp = 1 # yes
        #     else:
        #         VPrevalentHyp = 0 # no
        #     VPrevalentHyp = int(VPrevalentHyp)
            		 		

    except:
        second.destroy()

        third = tk.Tk()
        third.title("Failure!")
        lb = tk.Label(third, text="Some entries are empty or have an incorrect input.\n Please fill out all boxes with a correct input!\n All boxes need to be filled out.")
        sLeft   =  "%s" % 650   
        sTop    =  "%s" % 450   
        sWidth  =  "%s" % 600   
        sHeight =  "%s" % 100   
        lb.pack()

        third.wm_geometry(sWidth+"x"+sHeight+"+"+sLeft+"+"+sTop)
        third.resizable(width=0, height=0) # Verhinderung, dass die Fenstergröße verändert werden kann
        Button_try_again = tk.Button(third,text="Try again", bd=1, highlightthickness=0, command= third.destroy)
        Button_try_again.pack()
        
        third.mainloop()

    ## Creation of an array of all entries from the GUI
    xSubmit = np.array([VSex,VAge,VCSmoker,
                        VDiabetes,VTotChol,VSysBP,VDiaBP
                        ])
    
    
    second.title("Prediction of a ten year risk of coronary heart disease using risk stratification and framingham score!")

    ## Load the Model back from Github for classification
    # urllib.request.urlretrieve("https://raw.githubusercontent.com/Tobias149/FramingHam/main/Data%20science%20models/RandomForest_Model.pkl", "RandomForest_Model.pkl")
    # pickled_model_RF = pickle.load(open('RandomForest_Model.pkl', 'rb'))

    #def framingham_10year_risk(VAge, VSex, VCSmoker, VTotChol, VSysBP, VDiaBP, VDiabets):
    #intialize some things ----------
    points = 0 
    #percent_risk = StringVar()
    percent_risk = tk.IntVar(value=0)

    #Process females -----------------------------------------------------------
    if   VSex ==0:

        # VAge - VSex       
        if  30 <=   VAge   <= 34:
            points-=9
        if  35 <=   VAge   <= 39:
            points-=4
        if  40 <=   VAge   <= 44:
            points-=0
        if  45 <=   VAge   <= 49:
            points+=3
        if  50 <=   VAge   <= 54:
            points+=6
        if  55 <=   VAge   <= 59:
            points+=7
        if  60 <=   VAge   <= 64:
            points+=8
        if  65 <=   VAge   <= 69:
            points+=8
        if  70 <=   VAge   <= 74:
            points+=8
        if  75 <=   VAge   <= 79:
            points+=8

        print(points)
        if   VTotChol   < 160:
          points-=2
        if 160<=  VTotChol  <=199:
          points-=0
        if 200<=  VTotChol  <=239:
          points+=1
        if 240<=  VTotChol  <=279:
          points+=1
        if   VTotChol   > 280:
          points+=3
        
        print(points)
        if   VSysBP   < 120:
           points-=3
        if 120<=  VSysBP  <=129:
           points-=0
        if 130<=  VSysBP  <=139:
           points-=0
        if 140<=  VSysBP  <=159:
           points+=2
        if  160< VSysBP:
           points+=3
        print(points)
        if   VDiaBP   < 80:
           points-=3
        if 80<=  VDiaBP  <=84:
           points-=0
        if 85<=  VDiaBP  <=89:
           points-=0
        if 90<=  VDiaBP  <=99:
           points+=2
        if   VDiaBP   >= 100:
           points+=3

        print(points)
        if   VDiabetes   ==0:
          points-=0
        else:
          points+=4
        
        print(points)
        if   VCSmoker   ==0:
          points-=0
        else:
          points+=2
        print(points)
         #calulate % risk for males
        if points <= -2:
            percent_risk = 1
        elif points == -1:
            percent_risk = 2
        
        elif points == 0:
            percent_risk = 2
            
        elif points == 1:
            percent_risk = 2
            
        elif points == 2:
            percent_risk = 2
            
        elif points == 3:
            percent_risk = 2
            
        elif points == 4:
            percent_risk = 2
            
        elif points == 5:
            percent_risk = 4 
            
        elif points == 6:
            percent_risk = 5
            
        elif points == 7:
            percent_risk = 6
            
        elif points == 8:
            percent_risk = 7
            
        elif points == 9:
            percent_risk = 8 
            
        elif points == 10:
            percent_risk = 10
            
        elif points == 11:
            percent_risk = 11

        elif points == 12:
            percent_risk = 12
            
        elif points == 13:
            percent_risk = 15
            
        elif points == 14:
            percent_risk = 18
            
        elif points == 15:
            percent_risk = 20

        elif points == 16:
            percent_risk = 24
        
        elif points >= 17:
            percent_risk = 27



    else:
       # VAge - VSex       
        if  30 <=   VAge   <= 34:
            points-=1
        if  35 <=   VAge   <= 39:
            points-=0
        if  40 <=   VAge   <= 44:
            points+=1
        if  45 <=   VAge   <= 49:
            points+=2
        if  50 <=   VAge   <= 54:
            points+=3
        if  55 <=   VAge   <= 59:
            points+=4
        if  60 <=   VAge   <= 64:
            points+=5
        if  65 <=   VAge   <= 69:
            points+=6
        if  70 <=   VAge   <= 74:
            points+=7
        if  75 <=   VAge   <= 79:
            points+=8


        if   VTotChol   < 160:
          points-=3
        if 160<=  VTotChol  <=199:
          points-=0
        if 200<=  VTotChol  <=239:
          points+=1
        if 240<=  VTotChol  <=279:
          points+=2
        if   VTotChol   > 280:
          points+=3
        

        if   VSysBP   < 120:
           points-=0
        if 120<=  VSysBP  <=129:
           points-=0
        if 130<=  VSysBP  <=139:
           points+=1
        if 140<=  VSysBP  <=159:
           points+=2
        if 160 < VSysBP:
           points+=3


        if   VDiaBP   < 80:
           points-=0
        if 80<=  VDiaBP  <=84:
           points-=0
        if 85<=  VDiaBP  <=89:
           points+=1
        if 90<=  VDiaBP  <=99:
           points+=2
        if   VDiaBP   >= 100:
           points+=3


        if   VDiabetes   ==0:
          points-=0
        else:
          points+=2
        

        if   VCSmoker   ==0:
          points-=0
        else:
          points+=2


        #calulate % risk for males
        
        if points <= -1:
            percent_risk = 2
        
        elif points == 0:
            percent_risk = 3
            
        elif points == 1:
            percent_risk = 3
            
        elif points == 2:
            percent_risk = 4
            
        elif points == 3:
            percent_risk = 5
            
        elif points == 4:
            percent_risk = 7 
            
        elif points == 5:
            percent_risk = 8
            
        elif points == 6:
            percent_risk = 10
            
        elif points == 7:
            percent_risk = 13
            
        elif points == 8:
            percent_risk = 16
            
        elif points == 9:
            percent_risk = 20
            
        elif points == 10:
            percent_risk = 25
            
        elif points == 11:
            percent_risk = 31

        elif points == 12:
            percent_risk = 37
            
        elif points == 13:
            percent_risk = 45
            
        elif points >= 14:
            percent_risk = 53


    print("VAge", VAge)
    print("VSex", VSex)
    print("VTotchol", VTotChol)
    print("VSysBP", VSysBP)
    print("VDiaBP", VDiaBP)
    print("VDiabetes", VDiabetes)
    print("VCSmoker", VCSmoker)
    print("percent_risk", percent_risk)
    print("points", points)
    
    if flag == 1:
        tk.Label(second, text="Rate of diagnosis with coronary heart disease within the next 10 years is:  " + str(percent_risk) + "%").pack() 
    else:
        #return percent_risk
        import pandas as pd
        #import numpy as np
        import sys, json
        df1 = pd.read_csv(r"C:\\Users\\nbounas\\Downloads\\my_data2.csv")
        #df1 = df1['male','prevalentHyp', 'diabetes', 'diaBP', 'sysBP', 'age', 'currentSmoker', 'TenYearCHD']
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import train_test_split
        import seaborn as sns
        import matplotlib.pyplot as plt
        X = df1.drop('TenYearCHD',1)
        y = df1.TenYearCHD
        X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.05, random_state=42)
        from imblearn.over_sampling import SMOTE
        smote = SMOTE()
        X_ros, y_ros = smote.fit_resample(X_train, y_train)
        ros_chd_plot=y_ros.value_counts().plot(kind='bar')
        #plt.show()
        skfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500, random_state=7)
        result = cross_val_score(mlp, X, y, cv = skfold, scoring='accuracy')
        print(result)
        #MLPClassifier=MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500, random_state=7)
        mlp.fit(X_ros.values, y_ros.values)
        y_pred=mlp.predict(X_test.values)
        print(y_pred)
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        def my_confusion_matrix(y_test, y_pred, plt_title, accuracy_title):
            cm=confusion_matrix(y_test, y_pred)
            print(f'{accuracy_title} accuracy score:', '{:.2%}'.format(accuracy_score(y_test, y_pred)))
            print(classification_report(y_test, y_pred))
            sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='Blues')
            plt.xlabel('Predicted Values')
            plt.ylabel('Actual Values')
            plt.title(plt_title)
            #plt.show()
            return cm
        print(my_confusion_matrix(y_test, y_pred, 'mlp CM', 'Accuracy with Mlp:'))
        #xSubmit = np.array([[1, 40, 1, 1, 190, 105, 80]])
        New_classification_predict = xSubmit.reshape(1,-1)
        print(New_classification_predict)
        prediction = mlp.predict(New_classification_predict)
        print("Prediction: {}".format(prediction))
        prediction = prediction[0]  # transfrom the value of the array into string

        if prediction == 1:    
            lb = tk.Label(second, text="Bad news: Your entries are classfied with: 1\n\n You will suffer from coronary heart disease in the next ten years!\n\n Please change your current lifestyle.")
            lb.pack()  
        else:
            lb = tk.Label(second, text="Good news: Your entries are classfied with: 0\n\n You will not suffer from coronary heart disease in the next ten years!\n\n You can carry on with your lifestyle.")
            lb.pack()
    
    ## Limitation of the result window size and changes
    sLeft   =  "%s" % 600   
    sTop    =  "%s" % 350   
    sWidth  =  "%s" % 500   
    sHeight =  "%s" % 100   

    second.wm_geometry(sWidth+"x"+sHeight+"+"+sLeft+"+"+sTop)
    second.resizable(width=0, height=0) 
    second.mainloop()


    
## Main Window
fenster= tk.Tk()
fenster.title("10 YEAR CORONARY HEART DISEASE PREDICTION")
back_gnd = tk.Canvas(fenster)
back_gnd.pack(expand=True, fill='both')

# Read and open the Image from Desktop
path1 = 'heart-disease.jpg'
back_gnd_image = Image.open(path1)

## Reszie the Image
resize_back_gnd_image = back_gnd_image.resize((1100, 600))
back_gnd_image = ImageTk.PhotoImage(resize_back_gnd_image)

back_gnd.create_image(0,0, anchor='nw', image= back_gnd_image)

# lbl1 = tk.Label(fenster, text="Source of image: Healthline.")
# lbl1 ["font"] = "Courier 10"
# lbl1 ["bg"] = "#FFFFFF"
# lbl1.pack() # show label

lbl2 = tk.Label(fenster, text="Source: https://www.clickatlife.gr/your-life/story/24016")
lbl2 ["font"] = "Courier 10"
lbl2 ["bg"] = "#FFFFFF"
lbl2.pack() # show label

lbl3 = tk.Label(None, text="10 YEAR CORONARY HEART DISEASE PREDICTION")
lbl3 ["font"] = "Courier 20"
lbl3 ["bg"] = "#FFFFFF"
back_gnd.create_window(550,25, window=lbl3, anchor='center') # show label in background

## Naming of features left hand
lbl_model = tk.Label(fenster, text="Select model ", anchor="e")
back_gnd.create_window(190,100, window=lbl_model, anchor='center', width=230)
lbl_Sex = tk.Label(fenster, text="Gender ", anchor="e")
back_gnd.create_window(190,150, window=lbl_Sex, anchor='center', width=230)
lbl_VAge = tk.Label(fenster, text="Age ", anchor="e")
back_gnd.create_window(190,200, window=lbl_VAge, anchor='center', width=230)
lbl_CSmoker = tk.Label(fenster, text="Current Smoker ", anchor="e")
back_gnd.create_window(190,250, window=lbl_CSmoker, anchor='center', width=230)

## Naming of features right hand
lbl_VDiabetes = tk.Label(fenster, text="Diabetes ", anchor="w") 
back_gnd.create_window(900,100, window=lbl_VDiabetes, anchor='center', width=230)
lbl_VTotChol = tk.Label(fenster, text="Cholesterol [mg/dL] ", anchor="w")
back_gnd.create_window(900,150, window=lbl_VTotChol, anchor='center', width=230)
# only_fram = tk.Label(fenster, text="Used only for framingham score ", anchor="e")
# back_gnd.create_window(620,150, window=only_fram, anchor='center', width=200)
lbl_VSysBP = tk.Label(fenster, text="Systolic Blood Pressure [mmHg] ", anchor="w")
back_gnd.create_window(900,200, window=lbl_VSysBP, anchor='center', width=230)
lbl_VDiaBP = tk.Label(fenster, text="Diastolic Blood Pressure [mmHg] ", anchor="w")
back_gnd.create_window(900,250, window=lbl_VDiaBP, anchor='center', width=230)

frame_Model =tk.Frame(fenster)
Model = ["Neural Network","Framingham score"] 
li_Model = tk.Listbox(fenster, exportselection=0, height=0)
for i in Model:
    li_Model.insert("end", i)
back_gnd.create_window(355,100, window=li_Model, anchor='center', width=105)

frame_Sex =tk.Frame(fenster)
Gender = ["female","male"] # 1 = Female, 0 = Male
li_SEX = tk.Listbox(fenster, exportselection=0, height=0)
for i in Gender:
    li_SEX.insert("end", i)
back_gnd.create_window(335,150, window=li_SEX, anchor='center', width=60)

txt_AGE = tk.Entry(fenster)
txt_AGE.insert(0, "23") # years
back_gnd.create_window(335,200, window=txt_AGE, anchor='center', width=60) 


CSmoker = ["yes","no"]
li_CSmoker = tk.Listbox(exportselection=0,height=0)
for i in CSmoker:
    li_CSmoker.insert("end", i)
back_gnd.create_window(335,250, window=li_CSmoker, anchor='center', width=60)

### Input features right hand ###

VDiabetes = ["yes","no"]
li_Diabetes = tk.Listbox(fenster, exportselection=0, height=0)
for i in VDiabetes:
    li_Diabetes.insert("end", i)
back_gnd.create_window(755,100, window=li_Diabetes, anchor='center', width=60)

txt_TotChol = tk.Entry(fenster)
txt_TotChol.insert(0, "190")
back_gnd.create_window(755,150, window=txt_TotChol, anchor='center', width=60)

txt_SysBP = tk.Entry(fenster)
txt_SysBP.insert(0, "140") #mean systolic was >=140 mmHg
back_gnd.create_window(755,200, window=txt_SysBP, anchor='center', width=60)

txt_DiaBP = tk.Entry(fenster)
txt_DiaBP.insert(0, "90") #mean Diastolic >=90 mmHg
back_gnd.create_window(755,250, window=txt_DiaBP, anchor='center', width=60)

## Close button
cmd_button_e = tk.Button(None, text="Close", bd=1, highlightthickness=0, command= ende)
back_gnd.create_window(630,550, window=cmd_button_e, anchor='sw', width=150, height=50)


## Submit button
cmd_button_s = tk.Button(None, text="Submit", bd=1, highlightthickness=0, command= submit)
back_gnd.create_window(330,550, window=cmd_button_s, anchor='sw', width=150, height=50)

fLeft   =  "%s" % 500    
fTop    =  "%s" % 250    
fWidth  =  "%s" % 1100   
fHeight =  "%s" % 600    

## Limitation of main window size
fenster.wm_geometry(fWidth+"x"+fHeight+"+"+fLeft+"+"+fTop)
fenster.resizable(width=0, height=0) 

## Loop end
fenster.mainloop()