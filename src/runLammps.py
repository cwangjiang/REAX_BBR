import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ID", "-i", help="ID of the run",type=str)
parser.add_argument("--CORE", "-n", help="Number of cores",type=int)
parser.add_argument("--CG", "-cg", help="core groups",type=int)
parser.add_argument("--NC", "-nc", help="Number of C",type=int)
args = parser.parse_args()
ID = args.ID  # initial ID
CORE = args.CORE # number of cores
CG = args.CG
NC = args.NC

if CG==1:
	size = 4
	shift = 0
elif CG==2:
	size = 4
	shift = 4
elif CG==3:
	size = 4
	shift = 8
elif CG==4:
	size = 4
	shift = 12
elif CG==12:
	size = 8
	shift = 0
elif CG==34:
	size = 8
	shift = 8
elif CG==1234:
	size = 16
	shift = 0

# if CG==0:
# 	core0 = int(np.random.rand()*(8-2*CORE+1)) # randomly select starting core
# 	core1 = core0 + 2*CORE -1 # end core ID
# elif CG==1:
# 	core0 = int(np.random.rand()*(8-2*CORE+1))+8 # randomly select starting core
# 	core1 = core0 + 2*CORE -1 # end core ID

core0 = int(np.random.rand()*(size-2*CORE+1))+shift # randomly select starting core
core1 = core0 + 2*CORE -1 # end core ID

# NC = 13
if NC==5:
	elaps = 5
else:
	elaps = 5

# assign simulation ID to this initial configuration
file = open('in.ALKO2','r')
text = file.read()
text = text.replace('read_data	data.ALKO2  # coordinates','read_data	data'+str(ID)+'.ALKO2  # coordinates')
file.close()
file = open('in.ALKO2','w')
file.write(text)
file.close

file = open('inC.ALKO2','r')
text = file.read()
text = text.replace('read_data	data.ALKO2  # coordinates','read_data	data'+str(ID)+'.ALKO2  # coordinates')
file.close()
file = open('inC.ALKO2','w')
file.write(text)
file.close

# exit()

i = 1 # i is the number of iterations
step_eachrun = 100000 # number of steps for each interation
broken = False

# run the 1st round of simulation
script = 'mpirun --cpu-set '+str(core0)+'-'+str(core1)+' --use-hwthread-cpus -np '+str(CORE)+' lmp_gpu -in in.ALKO2'
os.system(script)

def check_break_5(s):
	for j in np.arange(4,len(s),1):
		if s[j][0]=='C':
			if len(s[j])==1:
				return 1  # only one C, break, return 1
			elif s[j][1] != '5':
				return 1 # C with others, like CH, or C3
			elif s[j][1] == '5':
				return 0 # C with others, like CH, or C3
		else:
			continue
	return 0

def check_break_12(s):
	for j in np.arange(4,len(s),1):
		if s[j][0]=='C':
			if len(s[j])==1:
				return 1  # only one C, break, return 1
			elif len(s[j])==2:
				return 1 # Cx, break return 1	
			elif s[j][1:3] != '12':
				return 1 # C with others, like CH, or C3
			elif s[j][1:3] == '12':
				return 0 # C with others, like CH, or C3
		else:
			continue
	return 0

def check_break_13(s):
	for j in np.arange(4,len(s),1):
		if s[j][0]=='C':
			if len(s[j])==1:
				return 1  # only one C, break, return 1
			elif len(s[j])==2:
				return 1 # Cx, break return 1	
			elif s[j][1:3] != '13':
				return 1 # C with others, like CH, or C3
			elif s[j][1:3] == '13':
				return 0 # C with others, like CH, or C3
		else:
			continue
	return 0

while 1:
	# check if alken is broken:
	file = open('species'+str(i-1)+'.ALKO2')
	text = file.read() # get all lammps script
	s = text.split('\n')
	ss_1 = s[-3].split() # get the last line
	ss_500 = s[int(-elaps*2-1)].split() # the last elaps'th line
 
# this is not good, since only one H can be disassociate, but alkane is still full
	# if int(ss[2]) > 2: # check if number of species >2, if so, alkane is broken
	# 	broken = True # ALK is breaking, stop everything
	# 	break

	# check if the last frame is breaking, if there is any Cx, and x != 6, if so, just exit
	# for j in np.arange(4,len(ss),1):
	# 	if ss[j][0]=='C':
	# 		if len(ss[j])==1:
	# 			exit()  # only one C
	# 		elif ss[j][1] != '5':
	# 			exit() # C with others, like CH, or C3
	# 	else:
	# 		continue

	if NC==5:
		if check_break_5(ss_1)&check_break_5(ss_500):
			# print(ss_1,ss_500)
			exit()
	elif NC==12:
		if check_break_12(ss_1)&check_break_12(ss_500):
			exit()
	elif NC==13:
		if check_break_13(ss_1)&check_break_13(ss_500):
			exit()
	# else:	# ALK is still not broken, continue next round of run
		## modify input run file
	file = open('inC.ALKO2') # continue run lammps template
	text = file.read()	
	text = text.replace('read_dump dumpC.reax.lammpstrj  x y z vx vy vz q','read_dump dumpC'+str(i-1)+'.reax.lammpstrj '+str(step_eachrun*i)+' x y z vx vy vz q')
	text = text.replace('1 all custom 10000 dumpC.reax.lammpstrj type id x y z vx vy vz q # for continue','1 all custom 10000 dumpC'+str(i)+'.reax.lammpstrj type id x y z vx vy vz q # for continue')
	text = text.replace('2 ALKC custom 10 dump.reax.lammpstrj type id x y z # for analysis','2 ALKC custom 10 dump'+str(i)+'.reax.lammpstrj type id x y z # for analysis')
	text = text.replace('fspecy all reaxff/species 10 10 100 species.ALKO2  ##10000frames=1Mb','fspecy all reaxff/species 10 10 100 species'+str(i)+'.ALKO2  ##10000frames=1Mb')
	
	with open('inC'+str(i)+'.ALKO2','w') as file: 
		file.write(text)

	## run new round of simulation
	script = 'mpirun --cpu-set '+str(core0)+'-'+str(core1)+' --use-hwthread-cpus -np '+str(CORE)+' lmp_gpu -in inC'+str(i)+'.ALKO2'
	os.system(script)

	i = i+1


