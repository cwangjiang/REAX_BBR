#### This code transfer the equilibrated annealing.gro file into a lammps input file data.CH4O2, but there is a input
# head.txt
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", help="input data link",type=str)
parser.add_argument("--ID", "-ID", help="input ID")

args = parser.parse_args()
inputlink = args.input
ID = args.ID
# print(link)
# width = int(args.width)

inputs = np.loadtxt(inputlink,comments='#',skiprows=0,dtype=str) # input data.

N=		int(inputs[0,2])# number of atoms
PrintV=	int(inputs[1,2]) #1: print V, 0, no V

filename1 = 'gros/eq'+str(ID)+'.gro'
f1 = open(filename1)
l = f1.readline()
l = f1.readline()

filename2 = 'head.txt'
f2 = open(filename2)
 # for python3


filename3 = 'datas/data'+str(ID)+'.ALKO2'
f3 = open(filename3,'w')


for i in range(16):
	l = f2.readline()
	print(l,file = f3,end='')
print('\n',file = f3)

L = [None]*6
V = [None]*N 

for i in range(N):
	l = f1.readline()
	cv = l.split() # I can't do auto splitting, since some coordinate value are connected.
	atom = cv[1][0]

	L[0] = '{:3d}'.format(i+1)

	if atom=='C':
		L[1] = '1'
	elif atom=='H':
		L[1] = '2'
	else:
		L[1] = '3'

	L[2] = '0.0'
	L[3] = '{:-7.2f}'.format(float(l[22:28])*10)
	L[4] = '{:-6.2f}'.format(float(l[30:36])*10)
	L[5] = '{:-6.2f}'.format(float(l[38:44])*10)

## velocity
	if False:
		V[i] = [None]*4

		V[i][0] = str(i+1)
		V[i][1] = '{:-.6f}'.format(float(l[45:52])/100)
		V[i][2] = '{:-.6f}'.format(float(l[53:60])/100)
		V[i][3] = '{:-.6f}'.format(float(l[61:68])/100)

	print(' '.join(L),file = f3)

if PrintV:
	print('\nVelocities \n',file = f3)

	for i in range(N):
		print(' '.join(V[i]),file = f3)

f1.close()
f2.close()
f3.close()

exit()
