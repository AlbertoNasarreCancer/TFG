import numpy as np 
import pandas as pd 
from music21 import converter, corpus, instrument, midi, note, chord, pitch
from music21 import stream
import music21

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from music21 import *
import os
import numpy as np
from xml.dom import minidom
from scipy.io import wavfile
import csv


def open_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()

    return midi.translate.midiFileToStream(mf)


def concat_path(path, child):
    return path + "/" + child


def extraer_datos_para_partitura(base_midi):

    d=[]
    b=[]
    a=[]
    notas=[]
    
    for n in base_midi.flat.notes: #in p
    
        if getattr(n, 'isNote', None) and n.isNote:
            
            notas.append(n)
            
            
            a.append(n.offset);
            b.append(n.duration.quarterLength);
            
            d.append(n.volume.velocity)
        if getattr(n, 'isChord', None) and n.isChord:
        
            i=0 

            
            for c in n:
                
                lala = note.Note(n.pitches[i].name + str(n.pitches[i].octave))
                
                notas.append(lala)
                

                
                a.append(n.offset);
                b.append(n.duration.quarterLength);
                d.append(n.volume.velocity)
                
                i=i+1
    return a,b,d,notas


def util_acordes(onsets):
	vector_bool_acordes=[]
	posicion_inicio=0
	posiciones_acordes=[]
	flag=False
	contador=0
	for n in range(0,len(onsets)):##llega a 250
		
		if (n!=len(onsets)-1 and n!=0):
		    
		    if (onsets[n]==onsets[n+1] or onsets[n]==onsets[n-1] ):
		        vector_bool_acordes.append(True)

		        if (onsets[n]==onsets[n+1] and onsets[n]!=onsets[n-1] ):
		            if (flag==False):
		                flag=True
		                posicion_inicio=n
		                
		        if (onsets[n]!=onsets[n+1] and onsets[n]==onsets[n-1] ):
		            if (flag):
		                flag=False
		                posiciones_acordes.append([posicion_inicio,n])
		    else:
		        vector_bool_acordes.append(False)
		else:
		    if(n==0):
		        if (onsets[n]==onsets[n+1]):###primero
		            vector_bool_acordes.append(True)
		            if (flag==False):
		                flag=True
		                posicion_inicio=n
		        else:
		            vector_bool_acordes.append(False)
		    else:
		        if (onsets[n]==onsets[n-1]):###ultimo
		            vector_bool_acordes.append(True)
		            if (flag):
		                flag=False
		                posiciones_acordes.append([posicion_inicio,n])
		        else:
		            vector_bool_acordes.append(False)
	return vector_bool_acordes,posiciones_acordes

def ordenar_acorde(acorde):  
    

    nota_grave_de_referencia=note.Note("C0")

    ints=[]
    for n in acorde:
        ints.append(interval.notesToChromatic(nota_grave_de_referencia, n).semitones)

    aaa=zip(*sorted(zip(ints, acorde)))

    return aaa[1]

def ordenar_acorde2(acorde,duraciones):  
    

    nota_grave_de_referencia=note.Note("C0")

    ints=[]
    for n in acorde:
        ints.append(interval.notesToChromatic(nota_grave_de_referencia, n).semitones)
    
    
    aaa=zip(*sorted(zip(ints, zip(acorde,duraciones))))

    return aaa[1]

def adjuntar(auxiliar):
    final=[]
    for n in auxiliar:
        
        final.append(n[1])
    return final

def ordenar(notas_vector,vector):

	for n in range(0,len(posiciones_acordes)):

		

		acorde=notas_vector[posiciones_acordes[n][0]:posiciones_acordes[n][1]+1]

		acorde_ordenado=ordenar_acorde(acorde)
		
		aux=vector[posiciones_acordes[n][0]:posiciones_acordes[n][1]+1]
		
		
		
		aux2=ordenar_acorde2(acorde,aux)
		
		#print(aux2)
		
		notas_vector[posiciones_acordes[n][0]:posiciones_acordes[n][1]+1]= acorde_ordenado
		
		
		vector[posiciones_acordes[n][0]:posiciones_acordes[n][1]+1]=adjuntar(aux2)

	return notas,vector 

def normalizar(vector):
    if hasattr(vector, "__len__"):
        normalizado=vector-(vector[0]*np.ones(len(vector)))
    else:
        normalizado=vector
    return normalizado


def calc_vector_posicion_compas():
	trozo1=onsets[0:20]                  #####intetar extraer cuando empieza un compas del xml de la partitura     En el stream se ven la clasves
	trozo2=onsets[20:24]
	trozo3=onsets[24:136]
	trozo4=onsets[137]
	trozo5=onsets[137:165]
	trozo6=onsets[165:181]
	trozo7=onsets[181:188]
	trozo8=onsets[188:212]
	trozo9=onsets[212:216]
	trozo10=onsets[216:242]
	trozo11=onsets[242:248]
	trozo12=onsets[248:251]

	trozos=[trozo1,trozo2,trozo3,trozo4,trozo5,trozo6,trozo7,trozo8,trozo9,trozo10,trozo11,trozo12]


	vector_posicion_compas_=[]
	for s in trozos:
		
		ss=normalizar(s)   
		#print(s)
		#print(ss)
		
		if hasattr(ss, "__len__"):
		    for n in ss:
		#print(np.mod(n,4))
		        if (np.mod(n,4)==0):
		            vector_posicion_compas_.append("FF")
		        else:
		            if (np.mod(n,4)==1 or np.mod(n,4)==2 or np.mod(n,4)==3 ):
		                vector_posicion_compas_.append("F")
		            else:
		                vector_posicion_compas_.append("W")
		else:
		    vector_posicion_compas_.append("FF")    

	return vector_posicion_compas_

def tones_to_tone(vector):
	tono=note.Note("G")
	vector_semitonos_to_tono=[]
	vector_is_consonat=[]

	for n in vector:
		intervalo=interval.Interval(tono, n)     #### o AL REVES?
		if (intervalo.semitones<0):
			vector_semitonos_to_tono.append(12+intervalo.semitones)
		else:
			vector_semitonos_to_tono.append(intervalo.semitones)     ####NORMALIZARLO?     TENER EN CUENTA LA ARMONIZACION DE LA MELODIA(ELIMINAR NOTAS ACORDES??)quedarme con la mas aguda?
		vector_is_consonat.append(intervalo.isConsonant())
	return vector_semitonos_to_tono,vector_is_consonat

def traducir_sin_octava(vector):
	final=[]
	for n in vector:
		la=note.Note(n.pitch.name)
		final.append(la)
	return final

def inter_ant_sig(notas):
    vector_int_ant_des=[]
    for i in range(0,len(notas)):
        if (i==0):

            int_anterior=interval.Interval(notas[i], notas[i])###################que pongo?

        else:
        
            int_anterior=interval.Interval(notas[i-1], notas[i])
    
        if (i==len(notas)-1):
            int_siguiente=interval.Interval(notas[i], notas[i])

        else:
            int_siguiente=interval.Interval(notas[i], notas[i+1])
    
        int_anterior_siguiente=[int_anterior.semitones,int_siguiente.semitones]  
    
        vector_int_ant_des.append(int_anterior_siguiente)
    return vector_int_ant_des

def calc_vect_ant_sig(vector):
	vector_dur_ant_des=[]
	for i in range(0,len(vector)):
		if (i==0):

		    dur_anterior=0###################que pongo?

		else:
		    
		    dur_anterior=vector[i-1]
		
		if (i==len(vector)-1):
		    
		    dur_siguiente=0

		else:
		    dur_siguiente=vector[i+1]
		
		dur_anterior_siguiente=[dur_anterior,dur_siguiente]  
		
		vector_dur_ant_des.append(dur_anterior_siguiente)
	return vector_dur_ant_des

def comparar_P(bef,next):

    if (bef>=7 and next<=5):
        return "IR"
    else:
        if(bef<=5 and next>=7):
            return"VP"
        else:
            return"P" 


def comparar_ID(bef,next):

    if (bef<=5 and next>=-3):
        return "IP"
    else:
        if(bef>=6 and next>=-3):
            return"R"
        else:
            if(bef>=6 and next<=-5):
                return "VR"
            else:
                return"ID"            
def narmour(vector_int_ant_des):
    vector_narmour=[]

    for n in range(0,len(vector_int_ant_des)):

        if (n==0 or n==len(vector_int_ant_des)-1):
   
            vector_narmour.append("null")

        

        else:

            before=vector_int_ant_des[n][0]    
            next=vector_int_ant_des[n][1]


            if (before==0 and next==0):

                vector_narmour.append("D")                 

                
            else:

                    if(before>0 and next>0):            
                        vector_narmour.append(comparar_P(before,next))
                    else:                    
                        if(before>0 and next<0):
                            vector_narmour.append(comparar_ID(before,next))
                        else:
                            vector_narmour.append("another")
    return(vector_narmour)


def aver(funcion):

	posiciones_to_set_false=[]

	for n in range(0,len(posiciones_acordes)):

		

		acorde=notas[posiciones_acordes[n][0]:posiciones_acordes[n][1]+1]

		acorde_ordenado=ordenar_acorde(acorde)
		nota_aguda=acorde_ordenado[-1]
		for s in range(posiciones_acordes[n][0],posiciones_acordes[n][1]+1):
		    if(interval.notesToChromatic(nota_aguda, notas[s]).semitones==0):
		        posiciones_to_set_false.append(s)



	####set false to the positions

	vector_bools_acorde_nuevo=vector_bool_acordes

	for n in posiciones_to_set_false:
		vector_bools_acorde_nuevo[n]=False    


	###extraer los trues(notas de armonizacion)
	vector_duraciones_melodia=[]
	vector_notas_melodia=[]
    
	i=0
	for n in vector_bools_acorde_nuevo:

		if (n==False):
		    vector_notas_melodia.append(notas[i])
		    vector_duraciones_melodia.append(notas[i].duration.quarterLength)
		i=i+1



	########
	vector_int_ant_des=inter_ant_sig(vector_notas_melodia)
    

    
	vector_narmour_aux=funcion(vector_int_ant_des) ######vector_narmour_aux=vector_int_ant_des ???  vector_narmour_aux=dur_ant_des  para hacer lo mismo con los otros vectores(centrarse en la melodia)
	vector_duraciones_aux=calc_vect_ant_sig(vector_duraciones_melodia)
    ########
	###anadir armonia

	vector_normur_armonia=[]
	vector_duraciones_armonia=[]
	i=0

	for n in vector_bools_acorde_nuevo:

		if (n==False):
		    vector_normur_armonia.append(vector_narmour_aux[i])
		    vector_duraciones_armonia.append(vector_duraciones_aux[i])
		    i=i+1
		else:
		    vector_normur_armonia.append("armonia")    
		    vector_duraciones_armonia.append("armonia")

	#print(vector_normur_armonia)
	#print(len(vector_normur_armonia))
	return vector_normur_armonia , vector_duraciones_armonia

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

def restar_listas(a,b):

    i = 0
    final=[]
    while i < len(a):
        final.append(60/(a[i]-b[i]))#mejor correlacion sin el 60
        i = i + 1
    return final

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


def rms_notas(vector):

	data_rms=[]

	for data in vector:
		
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


		vector_prueba=extraer_trozos_audio(derecho,vector_onset_offset)

		vector_rms=calcular_vector_rms(vector_prueba)

		data_rms.append(vector_rms)

		i=i+1
	return data_rms 


#inciar variables#

folderName = os.getcwd()



base_midi = open_midi(concat_path(folderName, "Partitura.mid"))

[onsets ,duraciones_partitura, volumen,notas]=extraer_datos_para_partitura(base_midi)



#sacar acordes#


[vector_bool_acordes,posiciones_acordes]=util_acordes(onsets)


#importante ORDENAR NOTAS MIDI#



[notas_ordenadas,duraciones_partitura_ordenadas]=ordenar(notas,duraciones_partitura)

[notas_ordenadas_2,onsets_ordenados]=ordenar(notas,onsets)

#feature ff f o w

vector_posicion_compas=calc_vector_posicion_compas()

#semitonos al tono y consonancia

[vector_semitonos_to_tono,vector_is_consonat]=tones_to_tone(traducir_sin_octava(notas_ordenadas))

#intervalo anterior y siguiente

vector_int_ant_des=inter_ant_sig(notas_ordenadas)


def funcionvacia(vector):

    return vector

intervalos_solo_melodia,vector_duraciones_armonia=aver(funcionvacia)
#aver(inter_ant_sig)
#duracion anerior y siguiente

vector_dur_ant_des=calc_vect_ant_sig(duraciones_partitura_ordenadas)

##############   Narmour structures

vector_narmour=narmour(vector_int_ant_des) 

vector_normur_armonia,vector_duraciones_armonia=aver(narmour)##funcion que se puede utilizar con otras funciones para comparar las notas qu no pertenecena a acordes

##############   pasar a csv ############################################################

notas_string=[]

for n in notas_ordenadas:
    notas_string.append(n.pitch.name)


vectores_intercalar=[notas_string,duraciones_partitura,onsets_ordenados,vector_posicion_compas,vector_semitonos_to_tono,vector_is_consonat,np.reshape(vector_int_ant_des,(251,2))[:,0],np.reshape(vector_int_ant_des,(251,2))[:,1],vector_bool_acordes,np.reshape(vector_dur_ant_des,(251,2))[:,0],np.reshape(vector_dur_ant_des,(251,2))[:,1],vector_normur_armonia]

vector_de_vectores=[]
for n in range(0,len(vectores_intercalar[0])):
    vector_intercalado=[]
    for s in vectores_intercalar:
        vector_intercalado.append(s[n])
    vector_de_vectores.append(vector_intercalado)
####escribir csv

with open('data.csv', mode='w') as data:
    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for n in vector_de_vectores:
        data_writer.writerow(n)

######################################################














########FEATURES INTERPRETES










base_midi1 = open_midi(concat_path(folderName, "MIDI Sujet1.mid"))
base_midi2 = open_midi(concat_path(folderName, "MIDI Sujet2.mid"))
base_midi3 = open_midi(concat_path(folderName, "MIDI Sujet3.mid"))
base_midi4 = open_midi(concat_path(folderName, "MIDI Sujet4.mid"))
base_midi5 = open_midi(concat_path(folderName, "MIDI Sujet5.mid"))
base_midi6 = open_midi(concat_path(folderName, "MIDI Sujet6.mid"))
base_midi7 = open_midi(concat_path(folderName, "MIDI Sujet7.mid"))
base_midi8 = open_midi(concat_path(folderName, "MIDI Sujet8.mid"))
base_midi9 = open_midi(concat_path(folderName, "MIDI Sujet9.mid"))


[onsets1 ,duraciones1, volumen1,notas1]=extraer_datos_para_partitura(base_midi1)
[onsets2 ,duraciones2 ,volumen2,notas2]=extraer_datos_para_partitura(base_midi2)
[onsets3 ,duraciones3 ,volumen3,notas3]=extraer_datos_para_partitura(base_midi3)
[onsets4 ,duraciones4 ,volumen4,notas4]=extraer_datos_para_partitura(base_midi4)
[onsets5 ,duraciones5 ,volumen5,notas5]=extraer_datos_para_partitura(base_midi5)
[onsets6 ,duraciones6 ,volumen6,notas6]=extraer_datos_para_partitura(base_midi6)
[onsets7 ,duraciones7 ,volumen7,notas7]=extraer_datos_para_partitura(base_midi7)
[onsets8 ,duraciones8 ,volumen8,notas8]=extraer_datos_para_partitura(base_midi8)
[onsets9 ,duraciones9 ,volumen9,notas9]=extraer_datos_para_partitura(base_midi9)


[parameters, frames, durations, labels, values]=extractSvlAnnotRegionFile("primertematempo.svl")
[parameters, frames2, durations, labels, values]=extractSvlAnnotRegionFile("segunda.svl")
[parameters, frames3, durations, labels, values]=extractSvlAnnotRegionFile("tercera.svl")
[parameters, frames4, durations, labels, values]=extractSvlAnnotRegionFile("cuarta.svl")
[parameters, frames5, durations, labels, values]=extractSvlAnnotRegionFile("quinta.svl")
[parameters2, frames6, durations2, labels2, values2]=extractSvlAnnotRegionFile("sexta.svl")
[parameters2, frames7, durations2, labels2, values2]=extractSvlAnnotRegionFile("septima.svl")
[parameters2, frames8, durations2, labels2, values2]=extractSvlAnnotRegionFile("octava.svl")
[parameters2, frames9, durations2, labels2, values2]=extractSvlAnnotRegionFile("novena.svl")




[notas1_ordenadas,duraciones1_ordenadas]=ordenar(notas1,duraciones1)
[notas_ordenadas_1,onsets1_ordenados]=ordenar(notas1,onsets1)

[notas2_ordenadas,duraciones2_ordenadas]=ordenar(notas2,duraciones2)
[notas_ordenadas_2,onsets2_ordenados]=ordenar(notas2,onsets2)

[notas3_ordenadas,duraciones3_ordenadas]=ordenar(notas3,duraciones3)
[notas_ordenadas_3,onsets3_ordenados]=ordenar(notas3,onsets3)

[notas4_ordenadas,duraciones4_ordenadas]=ordenar(notas4,duraciones4)
[notas_ordenadas_4,onsets4_ordenados]=ordenar(notas4,onsets4)

[notas5_ordenadas,duraciones5_ordenadas]=ordenar(notas5,duraciones5)
[notas_ordenadas_5,onsets5_ordenados]=ordenar(notas5,onsets5)

[notas6_ordenadas,duraciones6_ordenadas]=ordenar(notas6,duraciones6)
[notas_ordenadas_6,onsets6_ordenados]=ordenar(notas6,onsets6)

[notas7_ordenadas,duraciones7_ordenadas]=ordenar(notas7,duraciones7)
[notas_ordenadas_7,onsets7_ordenados]=ordenar(notas7,onsets7)

[notas8_ordenadas,duraciones8_ordenadas]=ordenar(notas8,duraciones8)
[notas_ordenadas_8,onsets8_ordenados]=ordenar(notas8,onsets8)

[notas9_ordenadas,duraciones9_ordenadas]=ordenar(notas9,duraciones9)
[notas_ordenadas_9,onsets9_ordenados]=ordenar(notas9,onsets9)
####
data_duraciones=[np.double(duraciones1_ordenadas),np.double(duraciones2_ordenadas),np.double(duraciones3_ordenadas),np.double(duraciones4_ordenadas),np.double(duraciones5_ordenadas),np.double(duraciones6_ordenadas),np.double(duraciones7_ordenadas),np.double(duraciones8_ordenadas),np.double(duraciones9_ordenadas)]

data_volumenes=[volumen1,volumen2,volumen3,volumen4,volumen5,volumen6,volumen7,volumen8,volumen9]

data_bpm=[calcular_bpm(frames), calcular_bpm(frames2),calcular_bpm(frames3),calcular_bpm(frames4),calcular_bpm(frames5),calcular_bpm(frames6),calcular_bpm(frames7),calcular_bpm(frames8),calcular_bpm(frames9)]

data_bpm_desviacion=[desviacion_bpm(data_bpm[0]),desviacion_bpm(data_bpm[1]),desviacion_bpm(data_bpm[2]),desviacion_bpm(data_bpm[3]),desviacion_bpm(data_bpm[4]),desviacion_bpm(data_bpm[5]),desviacion_bpm(data_bpm[6]),desviacion_bpm(data_bpm[7]),desviacion_bpm(data_bpm[8])]
###




notas_desviacion=[]

for n in range(0,9):
    notas_desviacion.append(calcular_desviaciones_por_nota(data_duraciones[n],data_bpm[n],duraciones_partitura))


plotear_arrays(notas_desviacion)
plotear_arrays(data_bpm_desviacion)




















########################            RMS                ##################################



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





data_rms=rms_notas(vector_data_wav)



plotear_arrays(data_rms)




##############   pasar a csv ############################################################

notas_string=[]

for n in notas_ordenadas:
    notas_string.append(n.pitch.name)

for n in range(len(intervalos_solo_melodia)):

    if(intervalos_solo_melodia[n]=="armonia"):
		intervalos_solo_melodia[n]=["armonia","armonia"]

for n in range(len(vector_duraciones_armonia)):

    if(vector_duraciones_armonia[n]=="armonia"):
		vector_duraciones_armonia[n]=["armonia","armonia"]

for x in range(0,len(notas_desviacion)):

	vectores_intercalar=[notas_string,duraciones_partitura,onsets_ordenados,vector_posicion_compas,vector_semitonos_to_tono,vector_is_consonat,np.reshape(intervalos_solo_melodia,(251,2))[:,0],np.reshape(intervalos_solo_melodia,(251,2))[:,1],vector_bool_acordes,np.reshape(vector_duraciones_armonia,(251,2))[:,0],np.reshape(vector_duraciones_armonia,(251,2))[:,1],vector_normur_armonia,notas_desviacion[x]]

	primera_vuelta=True
	vector_de_vectores=[]
	for n in range(0,len(vectores_intercalar[0])):
		vector_intercalado=[]
		for s in vectores_intercalar:
		    vector_intercalado.append(s[n])
		    if(primera_vuelta):
				vector_de_vectores.append(["notas_string","duraciones_partitura","onsets_ordenados","vector_posicion_compas","vector_semitonos_to_tono","vector_is_consonat","vector_int_ant_des0","vector_int_ant_des1","vector_bool_acordes","vector_dur_ant_des0","vector_dur_ant_des","vector_normur_armonia","notas_desviacion"])
				primera_vuelta=False
		vector_de_vectores.append(vector_intercalado)
	
	####escribir csv

	with open('data'+str(x)+'.csv', mode='w') as data:
		data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		for n in vector_de_vectores:
		    data_writer.writerow(n)

######################################################























###################  AVANZADO


import csv
vectorrr=[]
with open('proba.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
    	#print ', '.join(row)
        if (line_count !=0):
            vectorrr.append(np.double(row[2]))
        line_count = line_count +1
#print(line_count)
print(vectorrr)

duraciones_cambiadas=[]
i=0
for n in duraciones_partitura:
    duraciones_cambiadas.append(n*vectorrr[i])
    i=i+1
print(duraciones_cambiadas)
print(i)

#####################
i=0
s4 = stream.Stream()
for n in range(0,len(notas3_ordenadas)):

    if getattr(notas3_ordenadas[n], 'isNote', None) and notas3_ordenadas[n].isNote:
    


        notaa=notas3_ordenadas[n]
        #notaa.duration.quarterLength=0.25
        #notaa.duration.quarterLength=0.25
        #print(notaa.volume.velocity)
        
        s4.insert(onsets3_ordenados[n],notaa)
        i=i+1
        #print(onsets3_ordenados[n])
        #print(notas3_ordenadas[n].offset)
        print("--------------------")
    #else:

        #print(base_midi.flat[n])

        #s4.append(base_midi3.flat[n])

    if getattr(notas3_ordenadas[n], 'isChord', None) and notas3_ordenadas[n].isChord:


        s4.insert(notas3_ordenadas[n].offset,notas3_ordenadas[n])
        print("hola")
print(i)
mf4=midi.translate.streamToMidiFile(s4)

mf4.open(concat_path(folderName, "caca.mid"), 'wb')
mf4.write()
mf4.close()


