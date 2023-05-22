from functionalities import *

def main():
    X_train, X_test, y_train, y_test = preprocesar_datos()
    #Crear red neuronal
    #modeloRedNeuronal = construirRedNeuronalLSTM(X_train, y_train)

    CargarPesos()
    #print(evaluarPrecisionModelo(X_test, y_test, modeloRedNeuronal))

    #3
    colores = [(245,117,16), (117,245,16), (16,117,245)]
    # 1. New detection variables
    secuencia = []
    oracion = []
    predicciones = []
    umbralPrediccion = 0.7

    cap = cv2.VideoCapture(0)
    #Se establece al modelo de mediapipe
    with mp_modelo_holistico.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            #Leer los cuadros(imagenes) de la webcam, obtenemos
            # un valor de retorno y la imagen
            retorno, cuadro = cap.read()
            
            #Detección
            imagen, resultados = mediapipe_detection(cuadro, holistic)
          
            # Dibujo de los puntos de referencia
            draw_landmarks(imagen, resultados)
            # 2. Prediction logic
            puntosReferencia = extraer_puntos(resultados)
            secuencia.append(puntosReferencia)
            secuencia = secuencia[-30:]
        
            if len(secuencia) == 30:
                res = modeloRedNeuronal.predict(np.expand_dims(secuencia, axis=0))[0]
                print(acciones[np.argmax(res)])
                predicciones.append(np.argmax(res))
            
            
            #3. Viz logic
                if np.unique(predicciones[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > umbralPrediccion: 
                        
                        if len(oracion) > 0: 
                            if acciones[np.argmax(res)] != oracion[-1]:
                                oracion.append(acciones[np.argmax(res)])
                        else:
                            oracion.append(acciones[np.argmax(res)])

                if len(oracion) > 5: 
                    oracion = oracion[-5:]

                # Viz probabilities
                imagen = visualizar_probabilidades(res, acciones, imagen, colores)
                
            cv2.rectangle(imagen, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(imagen, ' '.join(oracion), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Show to screen
            #cv2.imshow('OpenCV Feed', imagen)

            #Transformamos el video al bits 
            suc, encode = cv2.imencode('.jpg', imagen)
            imagen = encode.tobytes()


            yield(b'--image\r\n'
                  b'content-Type: image/jpeg\r\n\r\n' + imagen + b'\r\n')
            
            #Romper el ciclo con la tecla q
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
        #Libera la webcam
        cap.release()
        #Cierra la ventana de mostrar video
        cv2.destroyAllWindows