##java -classpath weka.jar weka.classifiers.functions.MultilayerPerceptron -t "data2.csv" -T "data8.csv" -o
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def printArray(args):
    print ("\t".join(args))

archivos=["data0.csv","data1.csv","data2.csv","data3.csv","data4.csv","data5.csv","data6.csv","data7.csv","data8.csv"]

#cmd = 'java -classpath weka.jar weka.classifiers.functions.MultilayerPerceptron -t "data2.csv" -T "data8.csv" -o'

#weka.classifiers.trees.RandomForest

matriz_correlations=[]

for n in range(len(archivos)):
		correlations=[]
		for s in range(len(archivos)):

			cmd = 'java -classpath weka.jar weka.classifiers.trees.RandomForest -t "{}" -T "{}"  -o '.format(archivos[n],archivos[s])

			#returned_value = os.system(cmd)

			#print('returned value:', returned_value)


#weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0


			result = subprocess.check_output(cmd, shell=True)

			x=result.split("Correlation coefficient")

			"""
			print('result', x)

			print(len(x))
			print("-------------------")
			print(x[0])
			print("-------------------")
			print(x[1])
			print("-------------------")
			print(x[2])
			print("-------------------")
			"""

			y=x[2].split("Mean absolute error")

			resultado=float(y[0])
			correlations.append(resultado)
			#print(resultado)
		matriz_correlations = np.r_[matriz_correlations,correlations]

matriz_correlations_resize=np.resize(matriz_correlations,(int(np.sqrt(len(matriz_correlations))),int(np.sqrt(len(matriz_correlations)))))

for row in matriz_correlations_resize:
    printArray([str(np.float16(x)) for x in row])
print("-------------------")
print(np.mean(matriz_correlations_resize))

x = matriz_correlations
plt.hist(x, bins=20)
plt.ylabel('No of times')
plt.show()
