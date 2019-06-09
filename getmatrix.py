import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from music21 import converter, corpus, instrument, midi, note, chord, pitch
from music21 import stream
import music21
#music21.environment.set("graphicsPath", "/usr/bin/ristretto")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from music21 import *
import os
import numpy as np
from xml.dom import minidom
from scipy.io import wavfile

def open_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()

    return midi.translate.midiFileToStream(mf)


def concat_path(path, child):
    return path + "/" + child

def extraer_datos(base_midi):

    d=[]
    b=[]
    a=[]
    notas=[]
    for n in base_midi.flat.notes: #in p
    
        if getattr(n, 'isNote', None) and n.isNote:
            notas.append(n.pitch.name)
            a.append(n.offset);
            b.append(n.duration.quarterLength);
            #d.append(n.volume.getRealized())####
            d.append(n.volume.velocity)
        if getattr(n, 'isChord', None) and n.isChord:
        #notas.append("Chord")
            i=0 
            for c in n:
                notas.append(n.pitches[i].name)
                #print(n.pitches[0])
                a.append(n.offset);
                b.append(n.duration.quarterLength);
                d.append(n.volume.velocity)####
                #d.append(n.volume.getRealized())
                i=i+1
    return a,b,d,notas 

def restar_listas(a,b):

    i = 0
    final=[]
    while i < len(a):
        final.append(60/(a[i]-b[i]))#mejor correlacion sin el 60
        i = i + 1
    return final

def crear_vector_correlation(estatic,data):
    correlation_vector=[]
    for i in range(0,len(data)):
        #print(i)
        Covariance = np.cov(data[estatic], data[i], bias=True)[0][1] 
        correlation_coeficent=Covariance/(np.std(data[estatic])*np.std(data[i]))
        #print(correlation_coeficent)
        correlation_vector.append(correlation_coeficent)
        
    return correlation_vector



def crear_matriz(data):
    matriz_final=[]
    for i in range(0,len(data)):
        vector=crear_vector_correlation(i,data)
        matriz_final = np.r_[matriz_final,vector]
    return matriz_final


def calc_norm_factor(a,b):
    normalizar_factor=a[len(a)-1]/b[len(b)-1]
    return normalizar_factor

def printArray(args):
    print ("\t".join(args))


def extractSvlAnnotRegionFile(filename):
   
   
    dom = minidom.parse(filename)
    dataXML = dom.getElementsByTagName('data')[0]
    
   
    parametersXML = dataXML.getElementsByTagName('model')[0]
    
    
    datasetXML = dataXML.getElementsByTagName('dataset')[0]
    
    pointsXML = datasetXML.getElementsByTagName('point')
    parameters = {}
    for key in parametersXML.attributes.keys():
        parameters[key] = parametersXML.getAttribute(key)
    
    
    nbPoints = len(pointsXML)
    
    frames = np.zeros([nbPoints], dtype=np.float)
    durations = np.zeros([nbPoints], dtype=np.float)
    
    values = {}
    labels = {}
    
    
    for node in range(nbPoints):
        
        frames[node] = np.int(pointsXML[node].getAttribute('frame')) / np.double(parameters['sampleRate'])
       
        values[node] = pointsXML[node].getAttribute('value')
        labels[node] = pointsXML[node].getAttribute('label')
        
   
    return parameters, frames, durations, labels, values

def calcular_bpm(a):
    Lista1=np.delete(a, 0)
    Lista2=np.delete(a, a.shape[0] - 1)
    tiempos= restar_listas(Lista1,Lista2)
    return tiempos

def calcular_promedio(vector,longitud):
    total=0
    for n in range(0,longitud):
        total=total+(vector[n]/longitud)
    return total


def desviacion_bpm(vector):
    promedio=calcular_promedio(vector,6)######################################## jugar con este parametro######################
    desviacion_vector=[]
    for n in vector:
        desviacion_vector.append(n-promedio)
    return desviacion_vector

def calcular_desviaciones_por_nota(vector,data_bpm,duraciones_partitura):
    
    duracion_negra_referencia=calcular_promedio(data_bpm,6)/60
    vector_desviaciones_por_nota=[]
    for n in range(0,len(vector)):
        vector_desviaciones_por_nota.append(vector[n]/(duraciones_partitura[n]*duracion_negra_referencia))

    return vector_desviaciones_por_nota 


def plotear_arrays(vector):

    for n in range(0,len(vector)):
	    plt.plot(vector[n])

    plt.show()

def extraer_trozos_audio(cancion,vector):
    vector_audio_notas=[]
    for n in vector:
        vector_audio_notas.append(cancion[np.int(n[0]):np.int(n[1])])
    return vector_audio_notas

def calcular_vector_rms(vector):
    vector_rms=[]
    for n in range(0,len(vector)):
        #print(np.mean(np.square(vector[n])))
        rms = np.sqrt(np.abs(np.mean(np.square(vector[n]))))
        vector_rms.append(rms)
    return vector_rms    






folderName = os.getcwd()


#####################################
base_midi = open_midi(concat_path(folderName, "Partitura.mid"))
base_midi1 = open_midi(concat_path(folderName, "MIDI Sujet1.mid"))
base_midi2 = open_midi(concat_path(folderName, "MIDI Sujet2.mid"))
base_midi3 = open_midi(concat_path(folderName, "MIDI Sujet3.mid"))
base_midi4 = open_midi(concat_path(folderName, "MIDI Sujet4.mid"))
base_midi5 = open_midi(concat_path(folderName, "MIDI Sujet5.mid"))
base_midi6 = open_midi(concat_path(folderName, "MIDI Sujet6.mid"))
base_midi7 = open_midi(concat_path(folderName, "MIDI Sujet7.mid"))
base_midi8 = open_midi(concat_path(folderName, "MIDI Sujet8.mid"))
base_midi9 = open_midi(concat_path(folderName, "MIDI Sujet9.mid"))

[onsets ,duraciones_partitura, volumen,notas]=extraer_datos(base_midi)
[onsets1 ,duraciones1, volumen1,notas1]=extraer_datos(base_midi1)
[onsets2 ,duraciones2 ,volumen2,notas2]=extraer_datos(base_midi2)
[onsets3 ,duraciones3 ,volumen3,notas3]=extraer_datos(base_midi3)
[onsets4 ,duraciones4 ,volumen4,notas4]=extraer_datos(base_midi4)
[onsets5 ,duraciones5 ,volumen5,notas5]=extraer_datos(base_midi5)
[onsets6 ,duraciones6 ,volumen6,notas6]=extraer_datos(base_midi6)
[onsets7 ,duraciones7 ,volumen7,notas7]=extraer_datos(base_midi7)
[onsets8 ,duraciones8 ,volumen8,notas8]=extraer_datos(base_midi8)
[onsets9 ,duraciones9 ,volumen9,notas9]=extraer_datos(base_midi9)


[parameters, frames, durations, labels, values]=extractSvlAnnotRegionFile("primertematempo.svl")
[parameters, frames2, durations, labels, values]=extractSvlAnnotRegionFile("segunda.svl")
[parameters, frames3, durations, labels, values]=extractSvlAnnotRegionFile("tercera.svl")
[parameters, frames4, durations, labels, values]=extractSvlAnnotRegionFile("cuarta.svl")
[parameters, frames5, durations, labels, values]=extractSvlAnnotRegionFile("quinta.svl")
[parameters2, frames6, durations2, labels2, values2]=extractSvlAnnotRegionFile("sexta.svl")
[parameters2, frames7, durations2, labels2, values2]=extractSvlAnnotRegionFile("septima.svl")
[parameters2, frames8, durations2, labels2, values2]=extractSvlAnnotRegionFile("octava.svl")
[parameters2, frames9, durations2, labels2, values2]=extractSvlAnnotRegionFile("novena.svl")






#####################################

data_duraciones=[np.double(duraciones1),np.double(duraciones2),np.double(duraciones3),np.double(duraciones4),np.double(duraciones5),np.double(duraciones6),np.double(duraciones7),np.double(duraciones8),np.double(duraciones9)]

data_volumenes=[volumen1,volumen2,volumen3,volumen4,volumen5,volumen6,volumen7,volumen8,volumen9]

data_bpm=[calcular_bpm(frames), calcular_bpm(frames2),calcular_bpm(frames3),calcular_bpm(frames4),calcular_bpm(frames5),calcular_bpm(frames6),calcular_bpm(frames7),calcular_bpm(frames8),calcular_bpm(frames9)]

data_bpm_desviacion=[desviacion_bpm(data_bpm[0]),desviacion_bpm(data_bpm[1]),desviacion_bpm(data_bpm[2]),desviacion_bpm(data_bpm[3]),desviacion_bpm(data_bpm[4]),desviacion_bpm(data_bpm[5]),desviacion_bpm(data_bpm[6]),desviacion_bpm(data_bpm[7]),desviacion_bpm(data_bpm[8])]

#####################################

matriz_duraciones=crear_matriz(data_duraciones)

matriz_volumenes=crear_matriz(data_volumenes)

matriz_BPM=crear_matriz(data_bpm)

matriz_BPM_desviaciones=crear_matriz(data_bpm_desviacion)

##########################################

matriz_volumenes_resize=np.resize(matriz_volumenes,(int(np.sqrt(len(matriz_volumenes))),int(np.sqrt(len(matriz_volumenes)))))
matriz_duraciones_resize=np.resize(matriz_duraciones,(int(np.sqrt(len(matriz_duraciones))),int(np.sqrt(len(matriz_duraciones)))))
matriz_BPM_resize=np.resize(matriz_BPM,(int(np.sqrt(len(matriz_BPM))),int(np.sqrt(len(matriz_BPM)))))
matriz_BPM_desviaciones_resize=np.resize(matriz_BPM_desviaciones,(int(np.sqrt(len(matriz_BPM_desviaciones))),int(np.sqrt(len(matriz_BPM_desviaciones)))))

print("--------------------------------")
print("Matriz coeficientes de correlacion de los volumenes de cada nota")
print("--------------------------------")

for row in matriz_volumenes_resize:
    printArray([str(np.float16(x)) for x in row])

print("--------------------------------")
print("Matriz coeficientes de correlacion de las duraciones  de cada nota")
print("--------------------------------")

for row in matriz_duraciones_resize:
    printArray([str(np.float16(x)) for x in row]) 

print("--------------------------------")
print("Matriz coeficientes de correlacion de las curvas de BPM")
print("--------------------------------")

for row in matriz_BPM_resize:
    printArray([str(np.float16(x)) for x in row])

print("--------------------------------")
print("Matriz coeficientes de correlacion de las curvas de BPM respecto a las desviaciones sobre el bpm medio")
print("--------------------------------")

for row in matriz_BPM_desviaciones_resize:
    printArray([str(np.float16(x)) for x in row])
##############################################


plotear_arrays(data_bpm_desviacion)


###############################################








pruebas=[]

for n in range(0,9):
    pruebas.append(calcular_desviaciones_por_nota(data_duraciones[n],data_bpm[n],duraciones_partitura))



matriz_prueba=crear_matriz(pruebas)

matriz_prueba_resize=np.resize(matriz_prueba,(int(np.sqrt(len(matriz_prueba))),int(np.sqrt(len(matriz_prueba)))))

print("--------------------------------")
print("prueba")
print("--------------------------------")



for row in matriz_prueba_resize:
    printArray([str(np.float16(x)) for x in row])
####el plt bueno
###################################################


plotear_arrays(pruebas)

x = matriz_prueba
plt.hist(x, bins=20)
plt.ylabel('No of times')
plt.xlabel('Correlation')
plt.show()


#####################################################



#################################           RMS



data_onsets=[onsets1,onsets2,onsets3,onsets4,onsets5,onsets6,onsets7,onsets8,onsets9]

[fs,data_wav2]=wavfile.read(concat_path(folderName, "Sujet 2-limpio.wav"))
[fs,data_wav1]=wavfile.read(concat_path(folderName, "Sujet1.wav"))
[fs,data_wav3]=wavfile.read(concat_path(folderName, "Sujet3.wav"))
[fs,data_wav4]=wavfile.read(concat_path(folderName, "Sujet4.wav"))
[fs,data_wav5]=wavfile.read(concat_path(folderName, "Sujet5.wav"))
[fs,data_wav6]=wavfile.read(concat_path(folderName, "Sujet6.wav"))
[fs,data_wav7]=wavfile.read(concat_path(folderName, "Sujet7.wav"))
[fs,data_wav8]=wavfile.read(concat_path(folderName, "Sujet8.wav"))
[fs,data_wav9]=wavfile.read(concat_path(folderName, "Sujet9.wav"))

vector_data_wav=[data_wav1,data_wav2,data_wav3,data_wav4,data_wav5,data_wav6,data_wav7,data_wav8,data_wav9]

data_rms=[]


for data in vector_data_wav:
    
    if data.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif data.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    samples = data/ (max_nb_bit + 1.0) # samples is a numpy array of float representing the samples 

    izquierdo=samples[:,0].copy()
    derecho=samples[:,1].copy()

    #derecho=data[:,1].copy()


############funsion

    i=0
    vec_on=data_onsets[i]
    vec_dur=data_duraciones[i]
    duracion_audio=len(derecho)/fs
    normalizar_factor=duracion_audio/(vec_on[250]+vec_dur[250])  ##################+duracion[250]problema de exactitud con para normalizar,faltan datos hacer truco alargar ultima nota en hableton o recortar audios
    
    vector_onset_offset=[]

    

    for i in range(0,len(vec_on)):
       onset_offset=[vec_on[i]*fs*normalizar_factor,(vec_on[i]+vec_dur[i])*fs*normalizar_factor]
       vector_onset_offset.append(onset_offset)

############funsion
    #print(vector_onset_offset)
    #print("----------------------------------------------------")
    #print(len(derecho))
    vector_prueba=extraer_trozos_audio(derecho,vector_onset_offset)

    vector_rms=calcular_vector_rms(vector_prueba)

    data_rms.append(vector_rms)

    i=i+1

matriz_rms=crear_matriz(data_rms)

matriz_rms_resize=np.resize(matriz_rms,(int(np.sqrt(len(matriz_rms))),int(np.sqrt(len(matriz_rms)))))

print("--------------------------------")
print("Matriz coeficientes de correlacion de RMS")
print("--------------------------------")

for row in matriz_rms_resize:
    printArray([str(np.float16(x)) for x in row])



###################################################


#plotear_arrays(data_rms)


#####################################################



def desviacion_rms_rpmedio(vector):
    promedio=np.mean(vector)######################################## jugar con este parametro######################
    desviacion_vector=[]
    for n in vector:
        desviacion_vector.append(n/promedio)
    return desviacion_vector


vectores_rms_normalizados=[desviacion_rms_rpmedio(data_rms[0]),desviacion_rms_rpmedio(data_rms[1]),desviacion_rms_rpmedio(data_rms[2]),desviacion_rms_rpmedio(data_rms[3]),desviacion_rms_rpmedio(data_rms[4]),desviacion_rms_rpmedio(data_rms[5]),desviacion_rms_rpmedio(data_rms[6]),desviacion_rms_rpmedio(data_rms[7]),desviacion_rms_rpmedio(data_rms[8]),]



matriz_rms=crear_matriz(vectores_rms_normalizados)

matriz_rms_resize=np.resize(matriz_rms,(int(np.sqrt(len(matriz_rms))),int(np.sqrt(len(matriz_rms)))))

print("--------------------------------")
print("Matriz coeficientes de correlacion de RMS respecto al promedio")
print("--------------------------------")

for row in matriz_rms_resize:
    printArray([str(np.float16(x)) for x in row])

print(np.mean(matriz_rms_resize))
print(np.mean(matriz_volumenes_resize))
print(np.mean(matriz_prueba_resize))

###################################################


plotear_arrays(vectores_rms_normalizados)


#####################################################


plotear_arrays(data_volumenes)
   
