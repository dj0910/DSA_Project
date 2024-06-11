import pandas as pd
import csv
import numpy as np
#from ydata_profiling import ProfileReport

'''
Es gibt 319795 Einträge und 18 Attribute in dem Heart_2020-Datensatz. 
Es wird überprüft, ob ein Eintrag Nullwerte oder falsche Einträge enthält
Wenn fehlende Daten enthalten sind, wird eine Exception ausgegeben und man kann (vorerst manuell, da wir bisher nur 2 CSVs haben) überprüfen wo genau fehlende Daten sind.
TODO: Bias herausfiltern, Abwägung wie wichtig welche Attribute sind
'''

data_survey = pd.read_csv("resources\data_dirty\heart_2020.csv")

#Grober Überblick über die Daten
print(data_survey.head())
print(data_survey.dtypes)

#Check, ob die Datensätze alle vollständig ausgefüllt sind und nicht ein Attribut in einem Eintrag fehlt
data_survey_counted = data_survey.count()
if(data_survey_counted.nunique() != 1):
    #print("Hier ist ein Fehler")
    raise Exception("Daten sind nicht geeinget")

#Überprüfen, ob irgendwie ein na oder nan im Datensatz vorliegt, falls ja werde diese entfernt
data_survey_dropped_na = data_survey.dropna(how="any")

#Tage werden als int gespeichert, um spätere Darstellung besser zu veranschaulichen
data_survey_dropped_na["PhysicalHealth"] = data_survey_dropped_na["PhysicalHealth"].astype(int) 
data_survey_dropped_na["MentalHealth"] = data_survey_dropped_na["MentalHealth"].astype(int)

#TODO: ALLE ENGLISCHSPRACHIGEN INFORMATIONEN AUS DEM DATENSATZ ÜBERSEZEN(WICHTIG FÜR GRAPHEN)

data_survey_dropped_na["HeartDisease"][data_survey_dropped_na["HeartDisease"] == "Yes"] = "Ja"
data_survey_dropped_na["HeartDisease"][data_survey_dropped_na["HeartDisease"] == "No"] = "Nein"
data_survey_dropped_na["Smoking"][data_survey_dropped_na["Smoking"] == "Yes"] = "Ja"
data_survey_dropped_na["Smoking"][data_survey_dropped_na["Smoking"] == "No"] = "Nein"
data_survey_dropped_na["AlcoholDrinking"][data_survey_dropped_na["AlcoholDrinking"] == "Yes"] = "Ja"
data_survey_dropped_na["AlcoholDrinking"][data_survey_dropped_na["AlcoholDrinking"] == "No"] = "Nein"
data_survey_dropped_na["Stroke"][data_survey_dropped_na["Stroke"] == "Yes"] = "Ja"
data_survey_dropped_na["Stroke"][data_survey_dropped_na["Stroke"] == "No"] = "Nein"
data_survey_dropped_na["DiffWalking"][data_survey_dropped_na["DiffWalking"] == "Yes"] = "Ja"
data_survey_dropped_na["DiffWalking"][data_survey_dropped_na["DiffWalking"] == "No"] = "Nein"
data_survey_dropped_na["PhysicalActivity"][data_survey_dropped_na["PhysicalActivity"] == "Yes"] = "Ja"
data_survey_dropped_na["PhysicalActivity"][data_survey_dropped_na["PhysicalActivity"] == "No"] = "Nein"
data_survey_dropped_na["GenHealth"][data_survey_dropped_na["GenHealth"] == "Yes"] = "Ja"
data_survey_dropped_na["GenHealth"][data_survey_dropped_na["GenHealth"] == "No"] = "Nein"
data_survey_dropped_na["Asthma"][data_survey_dropped_na["Asthma"] == "Yes"] = "Ja"
data_survey_dropped_na["Asthma"][data_survey_dropped_na["Asthma"] == "No"] = "Nein"
data_survey_dropped_na["KidneyDisease"][data_survey_dropped_na["KidneyDisease"] == "Yes"] = "Ja"
data_survey_dropped_na["KidneyDisease"][data_survey_dropped_na["KidneyDisease"] == "No"] = "Nein"
data_survey_dropped_na["SkinCancer"][data_survey_dropped_na["SkinCancer"] == "Yes"] = "Ja"
data_survey_dropped_na["SkinCancer"][data_survey_dropped_na["SkinCancer"] == "No"] = "Nein"
data_survey_dropped_na["Sex"][data_survey_dropped_na["Sex"] == "Female"] = "Weiblich"
data_survey_dropped_na["Sex"][data_survey_dropped_na["Sex"] == "Male"] = "Männlich"
data_survey_dropped_na["AgeCategory"][data_survey_dropped_na["AgeCategory"] == "80 or older"] = "80 oder älter"
data_survey_dropped_na["Diabetic"][data_survey_dropped_na["Diabetic"] == "No"] = "Nein"
data_survey_dropped_na["Diabetic"][data_survey_dropped_na["Diabetic"] == "Yes"] = "Ja"
data_survey_dropped_na["Diabetic"][data_survey_dropped_na["Diabetic"] == "No, borderline diabetes"] = "Nein, aber Frühdiabetes"
data_survey_dropped_na["Diabetic"][data_survey_dropped_na["Diabetic"] == "Yes (during pregnancy)"] = "Ja, Schwangerschaftsdiabetes"
data_survey_dropped_na["Race"][data_survey_dropped_na["Race"] == "White"] = "Weiß"
data_survey_dropped_na["Race"][data_survey_dropped_na["Race"] == "Black"] = "Afroamerikanisch"
data_survey_dropped_na["Race"][data_survey_dropped_na["Race"] == "Asian"] = "Asiatisch"
data_survey_dropped_na["Race"][data_survey_dropped_na["Race"] == "Other"] = "Andere"
data_survey_dropped_na["Race"][data_survey_dropped_na["Race"] == "Hispanic"] = "Hispanisch"
data_survey_dropped_na["Race"][data_survey_dropped_na["Race"] == "American Indian/Alaskan Native"] = "Native American/Alaskan Native"
data_survey_dropped_na["GenHealth"][data_survey_dropped_na["GenHealth"] == "Excellent"] = "Exzellent"
data_survey_dropped_na["GenHealth"][data_survey_dropped_na["GenHealth"] == "Very good"] = "Sehr gut"
data_survey_dropped_na["GenHealth"][data_survey_dropped_na["GenHealth"] == "Good"] = "Gut"
data_survey_dropped_na["GenHealth"][data_survey_dropped_na["GenHealth"] == "Fair"] = "In Ordnung"
data_survey_dropped_na["GenHealth"][data_survey_dropped_na["GenHealth"] == "Poor"] = "Schlecht"


#Speicherung von Dataframe als csv in Ordner data_clean
data_survey_dropped_na.to_csv("resources\data_clean\heart_2020_clean.csv", index=False)

#profile_survey_scientific = ProfileReport(data_survey_dropped_na, title="Profiling Report")

'''
Es gibt 303 Einträge in dem heart_preditions-Datensatz mit 14 Attributen
TODO: Abwägung wie wichtig welche Attribute sind
'''

data_scientific = pd.read_csv("resources\data_dirty\heart_predictions.csv")

#Grober Überblick über die Daten
print(data_scientific.shape)
print(data_scientific.head())
print(data_scientific.dtypes)

#Filtere falsche Einträge aus dem Datensatz
data_scientific = data_scientific.drop(data_scientific[data_scientific["ca"] == 4].index)
data_scientific = data_scientific.drop(data_scientific[data_scientific["thal"] == 0].index)

#Nach dem Filtern von falschen Einträgen sind noch 296 Einträge im Datensatz enthalten
print(data_scientific.shape)

#Check, ob die Datensätze alle vollständig ausgefüllt sind und nicht ein Attribut in einem Eintrag fehlt
data_scientific_counted = data_scientific.count()
if(data_scientific_counted.nunique() != 1):
    #print("Hier ist ein Fehler")
    raise Exception("Daten sind nicht geeinget")

#Überprüfen, ob irgendwie ein na oder nan im Datensatz vorliegt, falls ja werde diese entfernt
data_scientific_dropped_na = data_scientific.dropna(how="any")

#Änderung der Spaltennamen, um ein besseres Verständnis für die Daten zu erhalten
data_scientific_renamed = data_scientific_dropped_na.rename(
    columns = {'cp':'chest_pain_type', 
               'trestbps':'resting_blood_pressure', 
               'chol': 'cholesterol',
               'fbs': 'fasting_blood_sugar',
               'restecg' : 'resting_electrocardiogram', 
               'thalach': 'max_heart_rate_achieved', 
               'exang': 'exercise_induced_angina',
               'oldpeak': 'st_depression', 
               'slope': 'st_slope', 
               'ca':'num_major_vessels', 
               'thal': 'thalassemia'}, 
    errors="raise")

#Einträge werden so angepasst, dass man sie besser nachvollziehen kann und es nicht kryptisch ist
#Dadurch bessere Wartbarkeit und besserer Überblick
data_scientific_renamed['sex'][data_scientific_renamed['sex'] == 0] = 'female'
data_scientific_renamed['sex'][data_scientific_renamed['sex'] == 1] = 'male'

data_scientific_renamed['chest_pain_type'][data_scientific_renamed['chest_pain_type'] == 0] = 'typical angina'
data_scientific_renamed['chest_pain_type'][data_scientific_renamed['chest_pain_type'] == 1] = 'atypical angina'
data_scientific_renamed['chest_pain_type'][data_scientific_renamed['chest_pain_type'] == 2] = 'non-anginal pain'
data_scientific_renamed['chest_pain_type'][data_scientific_renamed['chest_pain_type'] == 3] = 'asymptomatic'

data_scientific_renamed['fasting_blood_sugar'][data_scientific_renamed['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
data_scientific_renamed['fasting_blood_sugar'][data_scientific_renamed['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

data_scientific_renamed['resting_electrocardiogram'][data_scientific_renamed['resting_electrocardiogram'] == 0] = 'normal'
data_scientific_renamed['resting_electrocardiogram'][data_scientific_renamed['resting_electrocardiogram'] == 1] = 'ST-T wave abnormality'
data_scientific_renamed['resting_electrocardiogram'][data_scientific_renamed['resting_electrocardiogram'] == 2] = 'left ventricular hypertrophy'

data_scientific_renamed['exercise_induced_angina'][data_scientific_renamed['exercise_induced_angina'] == 0] = 'no'
data_scientific_renamed['exercise_induced_angina'][data_scientific_renamed['exercise_induced_angina'] == 1] = 'yes'

data_scientific_renamed['st_slope'][data_scientific_renamed['st_slope'] == 0] = 'upsloping'
data_scientific_renamed['st_slope'][data_scientific_renamed['st_slope'] == 1] = 'flat'
data_scientific_renamed['st_slope'][data_scientific_renamed['st_slope'] == 2] = 'downsloping'

data_scientific_renamed['thalassemia'][data_scientific_renamed['thalassemia'] == 1] = 'fixed defect'
data_scientific_renamed['thalassemia'][data_scientific_renamed['thalassemia'] == 2] = 'normal'
data_scientific_renamed['thalassemia'][data_scientific_renamed['thalassemia'] == 3] = 'reversable defect'

#Speicherung von Dataframe als csv in Ordner data_clean
data_scientific_renamed.to_csv("resources\data_clean\heart_predictions_clean.csv", index=False)

#profile_survey_scientific = ProfileReport(data_scientific_renamed, title="Profiling Report")
