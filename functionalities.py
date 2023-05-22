"""Importamos opencv para poder trabajar con la webcam"""
import cv2
""" Nos permite trabajar con arreglos especializados 
para el trato del conjunto de datos de nuestro modelo"""
import numpy as np
import os 
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.all_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_modelo_holistico = mp.solutions.holistic #Modelo hostic para la detección de los puntos de referencia 
mp_herraminetas_dibujado = mp.solutions.drawing_utils #Herramientas para dibujar el modelo

#Ruta para exportar los puntos extraídos en forma de numpy arrays
DATA_PATH = os.path.join('MP_Data')

#Directorio para tener registro del entrenamiento del modelo
log_dir = os.path.join('Logs')
#Creación del objeto "callback" de "tensorboard" el cual dara los datos obtenidos durante el entrenamiento del modelo 
tb_callback = TensorBoard(log_dir=log_dir)

#Las acciones que se quieren detectar
acciones = np.array(['hello', 'thanks', 'iloveyou'])
#Cantidad de videos o secuencias de imagenes conteniendo los datos
numeroSecuencias = 30
#Tamaño de los videos, en este caso de 30 cuadros o 30 imagenes
tamanioSecuencias = 30


#Creacion del modelo "Sequential API", para ordenar las capas de modelo de manera secuencial
modeloRedNeuronal = Sequential()

#Pasamos la imagen y el modelo holistico
def mediapipe_detection(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) #Convertimos de BGR a RGB
    imagen.flags.writeable = False #Hacer la imagen no editable
    resultados = modelo.process(imagen) #Hacer predicción
    imagen.flags.writeable = True #Hacer la imagen editable
    imagen = cv2.cvtColor(imagen,cv2.COLOR_RGB2BGR) #Convertir de RGB a BGR
    return imagen, resultados

def draw_landmarks(imagen, resultados):
    #Dibuja las conexiones de la cara
    mp_herraminetas_dibujado.draw_landmarks(imagen, resultados.face_landmarks, mp_modelo_holistico.FACEMESH_TESSELATION,
                              mp_herraminetas_dibujado.DrawingSpec(color=(80,110,10), thickness= 1, circle_radius=1),
                              mp_herraminetas_dibujado.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    #Dibuja las conexiones para la pose
    mp_herraminetas_dibujado.draw_landmarks(imagen, resultados.pose_landmarks, mp_modelo_holistico.POSE_CONNECTIONS, 
                              mp_herraminetas_dibujado.DrawingSpec(color=(80,110,10), thickness= 2, circle_radius=4),
                              mp_herraminetas_dibujado.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)) 
    #Dibuja las conexiones de la mano derecha
    mp_herraminetas_dibujado.draw_landmarks(imagen, resultados.left_hand_landmarks, mp_modelo_holistico.HAND_CONNECTIONS,
                              mp_herraminetas_dibujado.DrawingSpec(color=(121,22,76), thickness= 2, circle_radius=4),
                              mp_herraminetas_dibujado.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    #Dibuja las conexiones de la mando izquierda
    mp_herraminetas_dibujado.draw_landmarks(imagen, resultados.right_hand_landmarks, mp_modelo_holistico.HAND_CONNECTIONS,
                              mp_herraminetas_dibujado.DrawingSpec(color=(121,22,76), thickness= 2, circle_radius=4),
                              mp_herraminetas_dibujado.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 

def extraer_puntos(resultados):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in resultados.pose_landmarks.landmark]).flatten() if resultados.pose_landmarks else np.zeros(33*4)
    rostro = np.array([[res.x,res.y,res.z] for res in resultados.face_landmarks.landmark]).flatten() if resultados.face_landmarks else np.zeros(468*3)
    manoIzquierda = np.array([[res.x,res.y,res.z] for res in resultados.left_hand_landmarks.landmark]).flatten() if resultados.left_hand_landmarks else np.zeros(21*3)
    manoDerecha = np.array([[res.x,res.y,res.z] for res in resultados.right_hand_landmarks.landmark]).flatten() if resultados.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,rostro,manoIzquierda, manoDerecha])

def crear_directorio():
    for accion in acciones:
        for secuencia in range(numeroSecuencias):
            try:
                os.makedirs(os.path.join(DATA_PATH, accion, str(secuencia)))
            except:
                pass

def preprocesar_datos():
    #Donde se van a guardar todas las secuencias de cada una de las accione
    #Pudiendolas reconocer con etiquetas
    secuencias, etiquetas = [], []
    #Hacer un mapa de etiquetas de la forma "accion":numero_accion
    mapa_etiquetas = {label:num for num, label in enumerate(acciones)}
    for accion in acciones:
        for secuencia in np.array(os.listdir(os.path.join(DATA_PATH, accion))).astype(int):
            cuadros = []
            for numeroCuadros in range(tamanioSecuencias):
                #Agregar los cuadros(imagenes) de cada secuencia(video) en la variable res
                res = np.load(os.path.join(DATA_PATH, accion, str(secuencia), "{}.npy".format(numeroCuadros)))
                cuadros.append(res)
            secuencias.append(cuadros)
            etiquetas.append(mapa_etiquetas[accion])
    X = np.array(secuencias)
    y = to_categorical(etiquetas).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return X_train, X_test, y_train, y_test
    
def construirRedNeuronalLSTM(X_train, y_train):
    #Se añaden tres capas "LSTM", siendo el primer parametro la cantidad de unidades neuronales de la capa
    #el segundo el envio o no envio de las secuencias a la capa que le sigue.
    modeloRedNeuronal.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    modeloRedNeuronal.add(LSTM(128, return_sequences=True, activation='relu'))
    modeloRedNeuronal.add(LSTM(64, return_sequences=False, activation='relu'))
    
    #Se añaden las capas "Dense" que están conectadas completamente
    modeloRedNeuronal.add(Dense(64, activation='relu'))
    modeloRedNeuronal.add(Dense(32, activation='relu'))
    #Capa para las acciones, returnandonos la salida, cual accion es la más probable
    modeloRedNeuronal.add(Dense(acciones.shape[0], activation='softmax'))
    
    #Compilacion del modelo con el optimizador "Adam" para la reducción de errores modificando los atributos
    #de la red neuronal, además de la función 'categorical_crossentropy' para nuestro modelo de multiples clases de clasificación
    modeloRedNeuronal.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #Alimentamos el modelo
    modeloRedNeuronal.fit(X_train, y_train, epochs=300)
        
    #Guardar los "pesos" o los parametros que hacen nuestro modelo preciso 
    modeloRedNeuronal.save('action.h5')

def evaluarPrecisionModelo(X_test, y_test, model):
    yhat = modeloRedNeuronal.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)
    return accuracy_score(ytrue, yhat)

def CargarPesos():
    #Se añaden tres capas "LSTM", siendo el primer parametro la cantidad de unidades neuronales de la capa
    #el segundo el envio o no envio de las secuencias a la capa que le sigue.
    modeloRedNeuronal.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    modeloRedNeuronal.add(LSTM(128, return_sequences=True, activation='relu'))
    modeloRedNeuronal.add(LSTM(64, return_sequences=False, activation='relu'))
    
    #Se añaden las capas "Dense" que están conectadas completamente
    modeloRedNeuronal.add(Dense(64, activation='relu'))
    modeloRedNeuronal.add(Dense(32, activation='relu'))
    #Capa para las acciones, returnandonos la salida, cual accion es la más probable
    modeloRedNeuronal.add(Dense(acciones.shape[0], activation='softmax'))
    
    #Compilacion del modelo con el optimizador "Adam" para la reducción de errores modificando los atributos
    #de la red neuronal, además de la función 'categorical_crossentropy' para nuestro modelo de multiples clases de clasificación
    modeloRedNeuronal.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    modeloRedNeuronal.load_weights('action.h5')

def visualizar_probabilidades(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame