import streamlit as st
import pickle
from streamlit_option_menu import option_menu


selected=option_menu(
    menu_title = None ,
    options=["User","Technician"],
    icons=["person-fill","envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
) 

if selected=="User":
    
    st.title('Disease Prediction using symptoms')
    symp= ['','itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimple']
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

        model=pickle.load(open('newModel.pkl','rb'))
        pred=model.predict([new])

        st.write(f'### Disease might be {new_list[pred[0]]}')
        
    elif(button==True):
        st.error("Fill the Symptoms")

if selected=="Technician":
    with st.sidebar:
        option_menu(menu_title=None,
                    options=["Heart Disease","Brain Tumor","Malaria","Breast Cancer","Diabetes"],
                    )

