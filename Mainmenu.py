import tensorflow
from tensorflow import keras
from keras.models import load_model
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle

import cv2
from PIL import Image, ImageOps


selected=option_menu(
    menu_title = None ,
    options=["User","Technician"],
    icons=["person-fill","person-bounding-box"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
) 

if selected=="User":
    
    st.title('Disease Prediction using symptoms')
    symp= ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimple']
    dict={}
    i=0
    for j in symp :
        dict[j]=i
        i+=1

    st.subheader("What are the Symptoms: ")
    options = st.multiselect("",symp)

    button=st.button("Predict")

    if (button and options):
        list=[ i.lower() for i in options]
        new_list=[]

        for i in list :
            new_list.append(dict[i])


        new=[]
        for i in range(132) :
            if i in new_list :
                new.append(1)
            else :
                new.append(0)


        new_list=['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
            'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
            'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
            'Common Cold', 'Dengue', 'Diabetes ',
            'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
            'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
            'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
            'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
            'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
            'Osteoarthristis', 'Paralysis (brain hemorrhage)',
            'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
            'Typhoid', 'Urinary tract infection', 'Varicose veins',
            'hepatitis A']


        new_dict={}

        i=0
        for j in new_list :
            new_dict[j]=i
            i+=1

        model=pickle.load(open('models\Diseases\Disease_prediction_model.pkl','rb'))
        pred=model.predict([new])

        st.write(f'### Disease might be {new_list[pred[0]]}')
        
    elif(button==True):
        st.error("Fill the Symptoms")

if selected=="Technician":
    with st.sidebar:
       options=option_menu(menu_title=None,
                    options=["Heart Disease","Diabetes","Brain Tumor","Malaria","Breast Cancer"],
                    )
    if(options=="Heart Disease"):

        st.title('Heart Disease Prediction')
            
        age=st.number_input("Enter your Age",min_value=0)

        gender=st.radio("Gender",('Male','Female'),horizontal=True)

        if gender=='Male':
            sex=1
        else:
            sex=0

        cp=st.number_input("cp",min_value=0)
        trestbps=st.number_input("trestbps",min_value=0)
        chol=st.number_input("chol",min_value=0)
        fbs=st.number_input("fbs",min_value=0)
        restecg=st.number_input("restecg",min_value=0)
        thaclach=st.number_input("thaclach",min_value=0)
        exang=st.number_input("exang",min_value=0)
        oldpeak=st.number_input("oldpeak",min_value=0.0,format=f"%1.f")
        slope=st.number_input("slope",min_value=0)
        ca=st.number_input("ca",min_value=0)
        thal=st.number_input("thal",min_value=0)

        input_list=[age,sex,cp,trestbps,chol,fbs,restecg,thaclach,exang,oldpeak,slope,ca,thal]

        model = pickle.load(open('models/Heart/Heart_disease.pkl','rb'))


        if st.button("Result"):
            prediction=model.predict([input_list])
            
            if prediction[0]==1:
                st.error(""" ### High Probability of Heart Disease""")
            else:
                st.success(""" ### Huurayy! You are Fine""")
                st.balloons()

    if (options=="Diabetes"):

        st.title('Diabetes Prediction')

        age=st.number_input("Enter your Age",min_value=0)
        glucose=st.number_input("Enter your GlucoseLevel",min_value=0)
        bp=st.number_input("Enter your Blood Pressure",min_value=0)
        skinthick=st.number_input("Skin Thickness",min_value=0)
        bmi=st.number_input("BMI",min_value=0.0,format=f"%.1f",step=0.1)
        dpf=st.number_input("Diabetes Prediction Funtion",min_value=0.000,format=f"%.3f",step=0.001)
        insulin=st.number_input("Insulin Level",min_value=0,step=1)
        pregnant=st.number_input("No of Pregnancies",min_value=0,step=1)

        input_list=[age,glucose,bp,skinthick,bmi,dpf,insulin,pregnant]

        model = pickle.load(open('models/Diabetes/diabetes.pkl','rb'))

        if st.button("Result"):
            prediction=model.predict([input_list])           
            if prediction[0]==1:
                st.error(""" ### High Probability of Diabetes""")
            else:
                st.success("Huurayy! You are Fine")
                st.balloons()

    if (options=="Malaria"):
        st.title('Malaria Disease Prediction')

        class_names=["parasited","uninfected"]

        model=load_model("models\Malaria\malaria.h5")


        file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

        st.set_option('deprecation.showfileUploaderEncoding', False)

        def import_and_predict(image_data, model):

                size = (256,256)    
                image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
                image = np.asarray(image)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                img_reshape = img[np.newaxis,...]

                prediction = model.predict(img_reshape)
                return prediction

        if st.button("Predict"):
            if file is None:
                st.error("Please upload an image file")
            else:
                image = Image.open(file)
                prediction = import_and_predict(image, model)
                if prediction[0]<0.5 :
                    st.error(""" ### High probabilty of Malaria""")

                else :
                    st.success(""" ### Hurray you are Fine""")
                    st.balloons()
        if file:
            image = Image.open(file)
            width = st.slider('', 150, 500)
            st.image(image,width=width)


    if (options=="Brain Tumor"):
        st.title('Brain Tumor')

        class_names=["Benign","Malignant"]

        model=load_model("models\BrainTumor\BrainTumor.h5")
    
        file = st.file_uploader("Please upload scan file", type=["jpg", "png"])
    
        st.set_option('deprecation.showfileUploaderEncoding', False)
    
        def import_and_predict(image_data, model):
            
                size = (64,64)    
                image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
                image = np.asarray(image)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                img_reshape = img[np.newaxis,...]
            
                prediction = model.predict(img_reshape)
                return prediction
    
        if st.button("Predict"):    
            if file is None:
                st.error("Please upload an image file")
            else:
                image = Image.open(file)
                prediction = import_and_predict(image, model)
                pred=np.argmax(prediction)
                if pred==1 :
                    st.header("High probabilty of Brain Tumor")
                    
                else :
                    st.success("""### Hurray you are Fine""")
                    st.balloons()
        if file:
            image = Image.open(file)
            width = st.slider('', 150, 500)
            st.image(image,width=width)



    if (options=="Breast Cancer"):
        st.title('Breast Cancer Prediction')

        class_names=["Benign","Malignant"]

        model=load_model("models/BreastCancer/breast_cancer.h5")
    
        file = st.file_uploader("Please upload scan file", type=["jpg", "png"])
    
        st.set_option('deprecation.showfileUploaderEncoding', False)
    
        def import_and_predict(image_data, model):
            
                size = (256,256)    
                image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
                image = np.asarray(image)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                img_reshape = img[np.newaxis,...]
            
                prediction = model.predict(img_reshape)
                return prediction
    
        if st.button("Predict"):    
            if file is None:
                st.error("Please upload an image file")
            else:
                image = Image.open(file)
                prediction = import_and_predict(image, model)
                if prediction[0]>0.5 :
                    st.error(""" ### High probabilty of Breast Cancer""")
                    
                else :
                    st.success(""" ### Hurray you are Fine""")
                    st.balloons()

        if file:
            image = Image.open(file)
            width = st.slider('', 150, 500)
            st.image(image,width=width)
