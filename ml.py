#Import Stuff
from keras.preprocessing import sequence
from keras.preprocessing import text
import numpy as np
import tensorflow as tf
import pickle
import os

#Level 1 Model
def predict_l1(txt):
  model = tf.keras.models.load_model('Files//Models//Level1.model')
  mapping = ['Certificates', 'ComputerCertificate', 'Deploma', 'GraduationCertificate', 'HigherSecondaryCertificate', 'PostGraduationCertificate', 'SrSecondary', 'Applicationform', 'Checklist', 'CV', 'InterviewAssessmentForm', 'Joiningreport', 'MediclaimeCoverageForm', 'SalarySlip', 'Agreement', 'AppointmentLetter', 'CompensationStatement', 'ConfidentialInternalLetter', 'Confirmationletter', 'Employeeletter', 'GratuityFormF', 'IncrementLetter', 'Letterofintent', 'ManPowerRequestform', 'NETIXIS', 'OfferOfEmployment', 'Other', 'ReleivingLetter', 'TransferLetter', 'TransferRequestForm', 'BankAccountDetails', 'DrivingLicence', 'PanCard', 'Passport', 'VoterId']
  with open('Files//Tokenizer//Level1.pickle', 'rb') as handle:
      loaded_tokenizer = pickle.load(handle)
      seq= loaded_tokenizer.texts_to_sequences([txt])
      padded = sequence.pad_sequences(seq, maxlen=1000)
      pred = model.predict_classes(padded)
      return mapping[pred[0]]

#Level 2 Model
def predict_l2(txt):
  model = tf.keras.models.load_model('Files//Models//Level2.model')
  mapping = ['Certificates', 'Deploma', 'GraduationCertificate', 'HigherSecondaryCertificate', 'PostGraduationCertificate', 'SrSecondary', 'Applicationform', 'CV', 'InterviewAssessmentForm', 'Joiningreport', 'MediclaimeCoverageForm', 'SalarySlip', 'AppointmentLetter', 'CompensationStatement', 'ConfidentialInternalLetter', 'Confirmationletter', 'Employeeletter', 'GratuityFormF', 'IncrementLetter', 'ManPowerRequestform', 'OfferOfEmployment', 'Other', 'ReleivingLetter', 'TransferLetter', 'BankAccountDetails', 'DrivingLicence', 'PanCard', 'Passport', 'VoterId']
  with open('Files//Tokenizer//Level2.pickle', 'rb') as handle:
      loaded_tokenizer = pickle.load(handle)
      seq= loaded_tokenizer.texts_to_sequences([txt])
      padded = sequence.pad_sequences(seq, maxlen=1000)
      pred = model.predict_classes(padded)
      return mapping[pred[0]]

#Level 3 Model
def predict_l3(txt):
  model = tf.keras.models.load_model('Files//Models//Level3.model')
  mapping = ['Certificates', 'GraduationCertificate', 'HigherSecondaryCertificate', 'SrSecondary', 'Applicationform', 'CV', 'InterviewAssessmentForm', 'Joiningreport', 'MediclaimeCoverageForm', 'SalarySlip', 'AppointmentLetter', 'CompensationStatement', 'ConfidentialInternalLetter', 'Confirmationletter', 'Employeeletter', 'GratuityFormF', 'OfferOfEmployment', 'Other', 'ReleivingLetter', 'BankAccountDetails', 'DrivingLicence', 'PanCard']
  with open('Files//Tokenizer//Level3.pickle', 'rb') as handle:
      loaded_tokenizer = pickle.load(handle)
      seq= loaded_tokenizer.texts_to_sequences([txt])
      padded = sequence.pad_sequences(seq, maxlen=1000)
      pred = model.predict_classes(padded)
      return mapping[pred[0]]