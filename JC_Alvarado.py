##################################      LIBRERIAS

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import cv2  
import os  
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show
from skimage.color import rgb2gray
import time


##################################      OBJETOS
# Corazon: C     Diamante: D      Treboles: T      Picas: P
class Baraja:
    def __init__(self):
        self.mano = [['0','A'],['0','A'],['0','A'],['0','A'],['0','A']]
        self.Numeracion = 0

        self.Corazon = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.Diamante = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.Trebol = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.Pica = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.Contador_Cartas = 0
        
        # Valor de la mano
        self.Flor= False
        self.Escalera_Color= False
        self.Escalera= False
        self.Poker= False
        self.Full= False
        self.Color= False
        self.Escalera= False
        self.Tercia= False
        self.DoubleP= False
        self.Par= False
        self.Pachuca = False

        self.Parde = [0, 0]
        self.Terciade = 0
        self.Pokerde = 0

        #

# Resetea la mano pero no afecta memoria o el contador de cartas
    def ResetRonda(self):
        self.mano = [['0','A'],['0','A'],['0','A'],['0','A'],['0','A']]
        self.Numeracion = 0

        # Valor de la mano
        self.Flor= False
        self.Escalera_Color= False
        self.Escalera= False
        self.Poker= False
        self.Full= False
        self.Color= False
        self.Escalera= False
        self.Tercia= False
        self.DoubleP= False
        self.Par= False
        self.Pachuca = False
        
        self.Parde = [0, 0]
        self.Terciade = 0
        self.Pokerde = 0

# Primero hubica el palo, determina si ya salio, si ya salio = 1 manda trampa, si no la marca como que ya salio
    def Memoria(self, Numero, Palo):
        Numero = int(Numero)
        if Palo == 'C':
            if self.Corazon[Numero-1] == 0:
                self.Corazon[Numero-1] = 1
            else:
                print('Viejo mañoso, esa carta ya salio')
                Game = 0

        elif Palo == 'D':
            if self.Diamante[Numero-1] == 0:
                self.Diamante[Numero-1] = 1
            else:
                print('Viejo mañoso, esa carta ya salio')
                Game = 0

        elif Palo == 'P':
            if self.Pica[Numero-1] == 0:
                self.Pica[Numero-1] = 1
            else:
                print('Viejo mañoso, esa carta ya salio')
                Game = 0

        elif Palo == 'T':
            if self.Trebol[Numero-1] == 0:
                self.Trebol[Numero-1] = 1
            else:
                print('Viejo mañoso, esa carta ya salio')
                Game = 0

# Recibe n cantidad de cartas, y revisa espacio por espacio si hay cupo para la carta
# Si tiene espacio la recibe y hace memoria de si ya paso la carta
    # Num para numero y Pal para el palo
    def Dame(self,Ncards, modelo,baraja):
        for j in range(Ncards):
#            Recibe = input('¿Que carta me llego?_')       #Prueba con texto
            #Recibe = predecir(modelo)                     #Con fotos
            Recibe = leerCarta(baraja, modelo)
            #Recibe = Recibe.upper()                      
            Num,Pal =  Recibe.split()
            if self.mano[0] == ['0','A']:
                self.mano[0] = [Num,Pal]
                self.Memoria(self.mano[0][0],self.mano[0][1])

            elif self.mano[1] == ['0','A']:
                self.mano[1] = [Num,Pal]
                self.Memoria(self.mano[1][0],self.mano[1][1])

            elif self.mano[2] == ['0','A']:
                self.mano[2] = [Num,Pal]
                self.Memoria(self.mano[2][0],self.mano[2][1])

            elif self.mano[3] == ['0','A']:
                self.mano[3] = [Num,Pal]
                self.Memoria(self.mano[3][0],self.mano[3][1])

            elif self.mano[4] == ['0','A']:
                self.mano[4] = [Num,Pal]
                self.Memoria(self.mano[4][0],self.mano[4][1])
            else:
                print('ERROR: Ya no tengo espacio')
                Game = 0
            
# Logica de juego 
    def EsCorrida(self,order_list) :
    # Comprueba si el valor en la posicion i mas 1 es igual al siguiente valor
        for i in range(len(order_list)-1) :
            if ((order_list[i]+1) != (order_list[i+1])) :
                return False
        return True

    #logica de juego
    def Logica(self):
        self.Numeracion = [int(self.mano[0][0]),int(self.mano[1][0]),int(self.mano[2][0]),int(self.mano[3][0]),int(self.mano[4][0]),]
        Numeracion_ordenada = sorted(self.Numeracion)
        print (Numeracion_ordenada)

        # Color
        if self.mano[:][1] == self.mano[0][1]:
            Color = True

        # Escalera y flor
        if (Numeracion_ordenada == [1,10,11,12,13]) and (Color == True):
            Flor = True
        else:
            self.Escalera = self.EsCorrida(Numeracion_ordenada)

        # Escalera color 
        if (self.Color == True) and (self.Escalera == True):
            self.Escalera_Color = True

        # Poker, tercias y pares
        CuentaPares = 0
        for i in range(13):
            Contador = self.Numeracion.count(i+1)
            if Contador == 4:
                self.Poker = True
                self.Pokerde = i+1
            elif Contador == 3:
                self.Tercia = True
                self.Pokerde = i+1
            elif Contador == 2:
                self.Parde[CuentaPares] = i+1
                CuentaPares = CuentaPares + 1

        if (self.Tercia == True) and (CuentaPares == 1):
            self.Full = True
        elif CuentaPares == 2:
            self.DoubleP = True
        elif CuentaPares == 1:
            self.Par = True
        else:
            self.Pachuca = True

   # Decir que tienes
    def Cantala(self):
        if self.Flor == True:
            print('Flor imperial')
        elif self.Escalera_Color == True:
            print('Escalera de color')
        elif self.Poker == True:
            print('Poker')
        elif self.Full == True:
            print('Full')
        elif self.Color == True:
            print('Color')
        elif self.Escalera == True:
            print('Escalera')
        elif self.Tercia == True:
            print('Tercia')
        elif self.DoubleP == True:
            print('Dos pares')
        elif self.Par == True:
            print('Par')
        elif self.Pachuca == True:
            print('Pachuca imperial')

    # Cambiar cartas
    def MotorDeCambio(self,modelo,baraja):
        # Se le pregunta al oponente si va a cambiar
        print('¿Cambiarás cartas?')
        cuantas = int(input('¿Cuantas? '))

        #le toca a la comutadora decir si va a cambiar cartas o no
        print('Yo ...')

        # Saca de la mano n Cartas y pide (Llamar a Dame())
        if (self.Flor==True) or (self.Escalera_Color == True) or (self.Full == True) or (self.Poker == True) or (self.Escalera == True) or (self.Color == True):
            print('No cambio nada')

        elif (self.Tercia == True):
            print('Cambio 2')
            # revisa en las 5 posiciones cual no sirve
            for i in range(5):
                if int(self.mano[i][0]) != self.Terciade :
                    self.mano[i] = ['0','A']
            # Pide para llenar el hueco
            self.Dame(2,modelo,baraja)

        elif (self.DoubleP == True):
            print('Cambio 1')
            # revisa en las 5 posiciones cual no sirve
            for i in range(5):
                if (int(self.mano[i][0]) != self.Parde[0]) and (int(self.mano[i][0]) != self.Parde[1]) :
                    self.mano[i] = ['0','A']
            # Pide para llenar el hueco
            self.Dame(1,modelo,baraja)

        elif (self.Par == True):
            print('Cambio 3')
            for i in range(5):
                if int(self.mano[i][0]) != self.Parde[0] :
                    self.mano[i] = ['0','A']
            self.Dame(3,modelo,baraja)

        else:
            print('Cambio 4')
            for i in range(5):
                if int(self.mano[i][0]) != max(self.Numeracion) :
                    self.mano[i] = ['0','A']
            self.Dame(4,modelo,baraja)

##################################      Funciones
def Crupier(Jugador1,Jugador2):
    # JugadorN = [Flor, Escalera_Color, Poker, Full, Color, Escalera, Tercia, DoubleP, Par, Pachuca, Pokerde, Terciade, Parde, max(Numeracion)]
    # Jugador1 es la compu, Jugador2 es el oponente
    for R in range (10):
        if (Jugador1[R] == True) or (Jugador2[R] == True):
            # Flor
            if R == 0:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    print('Empate de Flores Imperiales')
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Escalera_color: Gana carta mas alta
            elif R == 1:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if Jugador1[13] == Jugador2[13]:
                        print('Empate de Escaleras de color')
                    elif Jugador1[13] > Jugador2[13]:
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Poker: Gana Poker mas alto
            elif R == 2:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if Jugador1[10] > Jugador2[10]:
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')
                
            # Full: Gana tercia mas grande
            elif R == 3:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if Jugador1[11] > Jugador2[11]:
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Color: Gana carta mas alta
            elif R == 4:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if Jugador1[13] == Jugador2[13]:
                        print('Empate de color')
                    elif Jugador1[13] > Jugador2[13]:
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Escalera: Gana carta mas alta
            elif R == 5:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if Jugador1[13] == Jugador2[13]:
                        print('Empate de escalera')
                    elif Jugador1[13] > Jugador2[13]:
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Tercia: Gana tercia mas alta
            elif R == 6:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if Jugador1[11] > Jugador2[11]:
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Dos Pares: Gana par mas alto 
            elif R == 7:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if max(Jugador1[12]) == max(Jugador2[12]):
                        # Si el primer par mas alto en ambos es igual gana el segundo par mas alto 
                        if min(Jugador1[12]) == min(Jugador2[12]):
                            # Si el segundo par es igual gana carta mas alta
                            if Jugador1[13] == Jugador2[13]:
                                print('Empate de Pares')
                            elif Jugador1[13] > Jugador2[13]:
                                print('JC Gana la ronda')
                            else:
                                print('Oponente Gana la ronda')
                        elif min(Jugador1[12]) > min(Jugador2[12]):
                            print('JC Gana la ronda')
                        else:
                            print('Oponente Gana la ronda')

                    elif max(Jugador1[12]) > max(Jugador2[12]):
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Par
            elif R == 8:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if max(Jugador1[12]) == max(Jugador2[12]):
                        # Si el par es igual gana carta mas alta
                        if Jugador1[13] == Jugador2[13]:
                            print('Empate de Pares')
                        elif Jugador1[13] > Jugador2[13]:
                            print('JC Gana la ronda')
                        else:
                            print('Oponente Gana la ronda')
                    elif max(Jugador1[12]) > max(Jugador2[12]):
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')

            # Pachuca: Gana carta mas alta
            else:
                if (Jugador1[R] == True) and (Jugador2[R] == True):
                    if Jugador1[13] == Jugador2[13]:
                        print('Empate de Pachucas Imperiales')
                    elif Jugador1[13] > Jugador2[13]:
                        print('JC Gana la ronda')
                    else:
                        print('Oponente Gana la ronda')
                    
                elif (Jugador1[R] == True):
                    print('JC Gana la ronda')
                else:
                    print('Oponente Gana la ronda')
            
            break

        #else:
        #   print('ERROR: Todos los valores de juego dan FALSE')

##################################
# Funciones de prediccion
def cargarModelo():
    modelo= tf.keras.models.load_model('modelo.h5')
    return modelo

def ConvertirImagen(imagen):
    img = rgb2gray(imagen)
    img = cv2.resize(img, (180, 180))
    img = img.reshape((1, img.shape[0], img.shape[1], 1))
    generator =  ImageDataGenerator(rotation_range=90, brightness_range=(0.5, 1.5), shear_range=15.0, zoom_range=[0.8, 1])
    generator.fit(img)
    iterator = generator.flow(img)
#    imgFinal = []
#    for i in range(16):
#        transformada = iterator.next()[0].astype('int')/255
#        transformada = img.reshape((1, img.shape[1], img.shape[2], 1))
#        imgFinal.append(transformada)
#    return imgFinal
    return img

def foto():
    cam = cv2.VideoCapture(0)
    time.sleep(3)
    s, img = cam.read()
    if s:
        cv2.namedWindow("cam-test")
        cv2.imshow("cam-test",img)
        cv2.waitKey(0)
        cv2.destroyWindow("cam-test")
        cv2.imwrite("imagen.jpg",img) 
    return imread('imagen.jpg')

def predecir(modelo):  # para que tome las cartas de las fotos
    df = pd.read_csv('card_labels.csv')
    labels=list(df)
    imagen = foto()
    trans = ConvertirImagen(imagen)
    pred = modelo.predict_classes(trans)
    print(labels[pred[0]])
    return labels[pred[0]]

def cargarBaraja():
    baraja=[]

    for i, img in tqdm(enumerate(os.listdir('baraja/'))):
        img = imread('baraja/'+img)
        baraja.append(img)
        
    shuffle(baraja)
    return baraja

def leerCarta(baraja, modelo):   # para que tome las barajas de los archivos
    df = pd.read_csv('card_labels.csv')
    labels=list(df)
    carta = baraja.pop(0)
    plt.figure()
    plt.imshow(carta)
    cartaProcesada = ConvertirImagen(carta)
    pred = modelo.predict_classes(cartaProcesada)
    print(labels[pred[0]])
    return labels[pred[0]]

##################################      ROOT
# Juego iniciado, Memoria en cero
Game = 1
Francesa = Baraja()
# Abrimos otra baraja para hacer mas sencillo la comparacion y usar los metodos de bajara
Oponente = Baraja()

modelo = cargarModelo()


while Game != 0:
    plt.close('all')
    baraja = cargarBaraja()
    print('A jugar') 
    
    # Visualizacion de control
    print(Francesa.mano)

    # Repartidor 
    Francesa.Dame(5,modelo,baraja)

    # Visualizacion de control
    print(Francesa.mano)
#    print(Francesa.Corazon)
#    print(Francesa.Diamante)
#    print(Francesa.Trebol)
#    print(Francesa.Pica)

    # Wamo a jugar a ver que show
    Francesa.Logica()

    # Visualizacion de control
    '''
    print(Francesa.Flor)
    print(Francesa.Escalera_Color)
    print(Francesa.Poker)
    print(Francesa.Full)
    print(Francesa.Color)
    print(Francesa.Escalera)
    print(Francesa.Tercia)
    print(Francesa.DoubleP)
    print(Francesa.Par)
    print(Francesa.Pachuca)
    '''
    Francesa.Cantala()

    # Cambio de cartas para ambos   
    Francesa.MotorDeCambio(modelo, baraja)

    # Visualizacion de control
    print(Francesa.mano)
    
    # Revisa de nuevo su propia mano para ver su juego
    Francesa.Logica()
    #Francesa.Cantala()

    # Hora de mostrar las manos 
    # Oponente muestra su mano y la computadora lo ve
    print('¿Que cartas tienes?')
    Oponente.Dame(5,modelo,baraja)
    # Deteminamos que tiene en la mano
    Oponente.Logica()
    # Oponente dice que tiene en la mano
    Oponente.Cantala()

    # Computadora dice que tiene en la mano
    Francesa.Cantala()

    # El crupier dice quien gana, primero ve las manos de...
    #Oponente
    JO = [Oponente.Flor, Oponente.Escalera_Color, Oponente.Poker, Oponente.Full, Oponente.Color, Oponente.Escalera, Oponente.Tercia, Oponente.DoubleP, Oponente.Par, Oponente.Pachuca, Oponente.Pokerde, Oponente.Terciade, Oponente.Parde, max(Oponente.Numeracion)]
    # Computadora
    JC = [Francesa.Flor, Francesa.Escalera_Color, Francesa.Poker, Francesa.Full, Francesa.Color, Francesa.Escalera, Francesa.Tercia, Francesa.DoubleP, Francesa.Par, Francesa.Pachuca, Francesa.Pokerde, Francesa.Terciade, Francesa.Parde, max(Francesa.Numeracion)]
    # Ahora si dice quien gaa 
    Crupier(JC,JO)

    Francesa.ResetRonda()
    Oponente.ResetRonda()
    Game = int(input ("Otra ronda? Si desea salir pulse 0: "))

print('Hasta luego.')