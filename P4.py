# Modelos Probabilísticos de Señales y Sistemas: Proyecto 4
# Elaborado por: Asael Guillén O. Carné: B63208

# 4.1 Modulación 16-QAM:

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

def fuente_info(imagen):

    img = Image.open(imagen)
    
    return np.array(img)

#------------------------------------------------------------------------------------------#

def rgb_a_bit(array_imagen):

    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

#------------------------------------------------------------------------------------------#

def modulador(bits, fc, mpp):

    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits
    '''
    Se crea una matriz de  n x 4 para agrupar paquetes de 4 bits.
    El "-1" se coloca ya que la cantiadad de filas es desconocida (son muchas).
    '''
    matriz_bits = bits.reshape(-1, 4)

    # 2. Construyendo un periodo de la señal portadora s(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp*2)  # muestras por periodo cada 2 bits
    portadoraI = np.cos(2*np.pi*fc*t_periodo) # Portadora en fase.
    portadoraQ = np.sin(2*np.pi*fc*t_periodo) # Portadora en cuadratura.
  

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # Señal moduladora de bits
 
    # 4. Asignar las formas de onda según los bits (16-QAM).

    i = 0
    for j, k, l, m in matriz_bits:
        '''
        Considere j = b1 y k = b2. Aquí realizamos las asignaciones para la portadora
        en fase. Es decir, se escoge el valor de A1 y se multiplica por el coseno. 
        '''
        if j == 0 and k == 0:
            senal_Tx[i*mpp : (i+2)*mpp] = portadoraI * -3
            moduladora[i*mpp : (i+2)*mpp] = -3
        elif j == 0 and k == 1:
            senal_Tx[i*mpp : (i+2)*mpp] = portadoraI * -1
            moduladora[i*mpp : (i+2)*mpp] = -1
        elif j == 1 and k == 1:
            senal_Tx[i*mpp : (i+2)*mpp] = portadoraI * 1
            moduladora[i*mpp : (i+2)*mpp] = 1
        elif j == 1 and k == 0:
            senal_Tx[i*mpp : (i+2)*mpp] = portadoraI * 3
            moduladora[i*mpp : (i+2)*mpp] = 3
        '''
        Considere l = b3 y m = b4. Aquí realizamos las asignaciones para la portadora
        en cuadratura. Es decir, se escoge el valor de A2 y se multiplica por el seno. 
        '''
        if l == 0 and m == 0:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadoraQ * 3
            moduladora[(i+2)*mpp : (i+4)*mpp] = 3
        elif l == 0 and m == 1:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadoraQ * 1
            moduladora[(i+2)*mpp : (i+4)*mpp] = 1
        elif l == 1 and m == 1:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadoraQ * -1
            moduladora[(i+2)*mpp : (i+4)*mpp] -1
        else:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadoraQ * -3
            moduladora[(i+2)*mpp : (i+4)*mpp] = -3
        i+=4

    # 5. Calcular la potencia promedio de la señal modulada
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
   
    return senal_Tx, Pm, portadoraQ, portadoraI, moduladora

#------------------------------------------------------------------------------------------#

def canal_ruidoso(senal_Tx, Pm, SNR):

    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

#------------------------------------------------------------------------------------------#

def demodulador(senal_Rx, portadorasin, portadoracos, mpp):
    
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)
    
    # Vector para la señal demodulada
    senal_demodulada = np.zeros(M)

    # Energía de un período de las portadoras
    EsI = np.sum(portadoraI**2)
    EsQ = np.sum(portadoraQ**2)


    # Demodulación:
    for i in range(0, N, 4): # Incrementos de 4 elementos iniciando con el primer elemento del array.

        # Producto interno de dos funciones para la portadora I
        productoI = senal_Rx[i*mpp:(i+2)*mpp] * portadoraI
        senal_demodulada[i*mpp:(i+2)*mpp] = productoI
        EpI = np.sum(productoI)

        '''
        Se encontró que el siguiente criterio de decisión es el adecuado para 
        obtener cero errores. El rango de valores por analizar es [-40, 40]
        '''

        # Criterio de decisión por detección de energía en portadora I

        if EpI > 40:
            bits_Rx[i] = 1      #b1
            bits_Rx[i+1] = 0    #b2 

        elif 0 < EpI < 40:
            bits_Rx[i] = 1      #b1
            bits_Rx[i+1] = 1    #b2

        elif -40 < EpI < 0:
            bits_Rx[i] = 0      #b1
            bits_Rx[i+1] = 1    #b2

        else:
            bits_Rx[i] = 0      #b1
            bits_Rx[i+1] = 0    #b2

        # Producto interno de dos funciones para la portadora Q
        productoQ = senal_Rx[(i+2)*mpp:(i+4)*mpp] * portadoraQ
        senal_demodulada[(i+2)*mpp:(i+4)*mpp] = productoQ
        EpQ = np.sum(productoQ)

        '''
        Se encontró que el siguiente criterio de decisión es el adecuado para 
        obtener cero errores. El rango de valores por analizar es [-40, 40]
        '''

        # Criterio de decisión por detección de energía en portadora Q
        if EpQ > 40:
            bits_Rx[i+2] = 0    #b3
            bits_Rx[i+3] = 0    #b4

        elif 0 < EpQ < 40:
            bits_Rx[i+2] = 0    #b3
            bits_Rx[i+3] = 1    #b4

        elif -40 < EpQ < 0:
            bits_Rx[i+2] = 1    #b3
            bits_Rx[i+3] = 1    #b4

        else:
            bits_Rx[i+2] = 1    #b3
            bits_Rx[i+3] = 0    #b4

 
    return bits_Rx.astype(int), senal_demodulada

#------------------------------------------------------------------------------------------#

def bits_a_rgb(bits_Rx, dimensiones):

    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

#------------------------------------------------------------------------------------------#

print("Simulación con modulador 16-QAM")

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 5   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)
print('Los bits son: ',bits_Tx)
print('El len de los bits es: ', len(bits_Tx))

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadoraQ, portadoraI, moduladora = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadoraQ, portadoraI, mpp)
print('El len de los bits es:', len(bits_Rx))

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

#------------------------------------------------------------------------------------------#

# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='c', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='m', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='y', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='k', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

###############################################################################################

# 4.2 Estacionaridad y ergocidad
import numpy as np
import matplotlib.pyplot as plt

# Tiempo para la muestra
t_m = np.linspace(0, 0.1,100)

# Posibles valores de A
A = [1, -1, 3, -3]

# Formas de onda
X_t = np.empty((8, len(t_m)))

plt.figure()

# Matriz con los valores de cada función posibles   
for i in A:
    x1 = i * np.cos(2*(np.pi)*fc*t_m) +  i*np.sin(2*(np.pi)*fc*t_m)
    x2 = -i * np.cos(2*(np.pi)*fc*t_m) +  i*np.sin(2*(np.pi)*fc*t_m) 
    X_t[i,:] = x1
    X_t[i+1,:] = x2
    plt.plot(t_m,x1, lw=2)
    plt.plot(t_m, x2, lw=2)    

# Promedio de las 8 realizaciones en cada instante 
P = [np.mean(X_t[:,i]) for i in range(len(t_m))]
plt.plot(t_m, P, lw=3,color='k',label='Promedio Realizaciones')

# Graficar el resultado teórico del valor esperado
E = np.mean(senal_Tx)*t_m  # Valor esperado de la señal 
plt.plot(t_m, E, '-.', lw=3,color='c',label='Valor teórico')

# Mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.legend()
plt.show()

###############################################################################################

# 4.3 Densidad espectral 

from scipy.fft import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2),color='b')
plt.xlim(0,20000)
plt.title("Densidad espectral")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.ion()
plt.show
plt.pause(0.001)

