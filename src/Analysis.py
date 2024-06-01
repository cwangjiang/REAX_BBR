import numpy as np 
import argparse
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt 
import matplotlib
import os
from numpy import linalg as LA
import FUNC.dmap as dp
from joblib import Parallel, delayed
import time
import multiprocessing

#
#
#
#
#

#######################
# input variables
#######################
#
#
#
#
#

# initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", help="input data link",type=str)
args = parser.parse_args()
inputlink = args.input
# print(link)
# width = int(args.width)

inputs = np.loadtxt(inputlink,comments='#',skiprows=0,dtype=str) 
# print(inputs)

path = inputs[0,2]
Combine = bool(int(inputs[1,2]))# number of frames
Ni = int(inputs[2,2])# number of independent initials
NC = int(inputs[3,2]) # number of carbon 
head = int(inputs[4,2]) # head lines of lammps trajectory file
chiral_atom1 = int(inputs[5,2])  # 2
chiral_atom2 = int(inputs[6,2])  # 3
L=int(inputs[7,2])  # box size 25 A
thresh = float(inputs[8,2])   #1.8 # bond thresh hold Anstrom
ellaps = int(inputs[9,2])  # ellapse steps for judging bond breaking, 1000 is okay
Ndmap = int(inputs[10,2]) # number of point to do dmap
ndmap = int(inputs[11,2])  # number of point from each initial to do dmap
calc_flag = int(inputs[12,2])
Nf = int(inputs[13,2]) # Nframe
scale = float(inputs[14,2])
dt = float(inputs[15,2]) # resolution to plot prob. dens.
deltat = float(inputs[16,2]) # resolution to search for best startime
dRg = float(inputs[17,2]) # resolutin of Rg
Maxstartime = float(inputs[18,2]) # maximum test startime
endtime = float(inputs[19,2]) # maximum time to cut
sim0 = int(inputs[20,2])
sim1 = int(inputs[21,2])
resample = int(inputs[22,2])
devec = float(inputs[23,2])
dmap_flag = int(inputs[24,2])
epsilon = float(inputs[25,2])
exponent = float(inputs[26,2])
h2tflag = int(inputs[27,2])
mirrorflag = int(inputs[28,2])

#
#
#
#
#
#
############################
# defined functions
############################
#
#
#
#
#
#

# function to load one dump trajectory to a 3-d array
def load_dump(filename,Nframe,NC,head):
	coord = np.zeros((Nframe,NC,3)) # coordinate template
	file = open(filename,'r')
	temp =np.zeros((NC,4)) # template for each frame
	for i in range(head+NC): # skip first 0 frame
		l=file.readline()
	for i in range(Nframe):  # loop over all frames
		for j in range(head): # skip head lines
			l=file.readline()
		for j in range(NC):   # loop over coordinates
			l=file.readline()
			s=l.split()
			# print(l)
			temp[j,:] = [int(s[1]),float(s[2]),float(s[3]),float(s[4])]
		order = np.argsort(temp[:,0]) # reorder the ID of each frame
		# print(temp[order,1:4])
		coord[i,:,:] = temp[order,1:4]
	return coord

# use the species file from lammps to find the breaking step/time
def find_breaking_time(filename):
	file = open(filename,'r') 
	text = file.read() # read all species text
	s = text.split('\n') # split in to lines
	n = int((len(s)-1)/2) # there is empty line at the last
	# print(n)
	alk = np.zeros(n) # store number of C of the 1st molecule
	for i in range(n): # loop over lines
		l1=s[i*2] # fine only species lines
		ss = l1.split() # split 
		if (ss[4][1]<='9')&(ss[4][1]>='0'):
			alk[i] = int(ss[4][1])# the 4th item is 1st molecue, find the number of C
		else:
			alk[i] = 1

	# print(alk)
	t = 0 # breaking step
	for i in range(n): # loop again to find breaking time
		if alk[i]==6: 
			continue # not breaking, continue
		elif alk[i+ellaps]!=6: # if current is breaking, and the ellaps steps later is still breaking, then breaking, but this is not accurate
			t=i #breaking step
			return t

def dist_PBC(x,y): # modiyf for PBC,compute euclidean disance between point x,y, take PBC into consideration
	delta = abs(x-y) #need to take abs
	select = delta>L/2
	delta[select] = L-delta[select] # take PBC into consideration
	d = np.sqrt(np.sum(delta**2))
	return d

def dist(x,y): # already PBC, complete, compute euclidean disance between point x,y, take PBC into consideration
	delta = abs(x-y) #need to take abs
	# select = delta>L/2
	# delta[select] = L-delta[select] # take PBC into consideration
	d = np.sqrt(np.sum(delta**2))
	return d

def dist_PBC_array(x,y): # already PBC, complete, compute euclidean disance between point x,y, take PBC into consideration
	delta = abs(x-y) #need to take abs
	select = delta>L/2
	delta[select] = L-delta[select] # take PBC into consideration
	d = np.sqrt(np.sum(delta**2,axis=2))
	return d

# obtain the bond length for chain 
def find_breaking_bond(I):
	if I%10==0:
		print(I)
	filename = 'dumps/dump'+str(I+1)+'.npy'
	coord = np.load(filename)
	# print(coord.shape)
	n = len(coord)
	bond_len = np.zeros((n,NC-1)) # chain with NC C atoms have NC-1 bonds
	for i in range(n):
		coordtemp = coord[i]
		for j in range(NC-1):
			bond_len[i,j] = dist_array(coordtemp[j,:],coordtemp[j+1,:]) # dist considered PBC

	# fiding breaking time based on bond length, this is more strict than species file, the time will be earlier
	# but not too differernt, and can be a verification
	# print(bond_len)
	t = 0
	breaking = np.sum(bond_len>thresh,axis=1) # if there is one bond > thresh, it's True, sum is true
	for i in range(n):
		# if i%1000==0:
			# print('n=',i)
		if breaking[i]==False: # not breaking
			continue
		elif (np.sum(breaking[i:(i+ellaps)])>ellaps*0.99): # if current and most the following ellaps frames are breaking, then it's breaking
			t=i 
			b = np.where(bond_len[i,:]>thresh) # find which bond is breaking
			bindex = b[0][0]
			# print(bond_len[i])
			# print(t)
			# return bond_len, coord[i], bindex, t # return the frame of breaking
			return coord[i], bindex, t 

	# print(bond_len.shape)
	# print(bond_len)
	
# obtain the bond length for chain 
def find_breaking_bond_array(I):
	if I%100==0:
		print(I)
	filename = 'dumps/dump'+str(I+1)+'.npy'
	coord = np.load(filename)
	# print(coord.shape)
	n = len(coord)
	bond_len = np.zeros((n,NC-1)) # chain with NC C atoms have NC-1 bonds

	bond_len = dist_PBC_array(coord[:,0:(NC-1),:],coord[:,1:NC,:])

	# for i in range(n):
	# 	coordtemp = coord[i]
	# 	for j in range(NC-1):
	# 		bond_len[i,j] = dist(coordtemp[j,:],coordtemp[j+1,:]) # dist considered PBC

	# fiding breaking time based on bond length, this is more strict than species file, the time will be earlier
	# but not too differernt, and can be a verification
	# print(bond_len)
	t = 0
	breaking = np.sum(bond_len>thresh,axis=1) # if there is one bond > thresh, it's True, sum is true
	for i in range(n):
		# if i%1000==0:
			# print('n=',i)
		if breaking[i]==False: # not breaking
			continue
		elif (np.sum(breaking[i:(i+ellaps)])>ellaps*0.99): # if current and most the following ellaps frames are breaking, then it's breaking
			t=i 
			b = np.where(bond_len[i,:]>thresh) # find which bond is breaking
			bindex = b[0][0]
			# print(bond_len[i])
			# print(t)
			# return bond_len, coord[i], bindex, t # return the frame of breaking
			return coord[i], bindex, t 

	# print(bond_len.shape)
	# print(bond_len)

def compute_chirality(v1,v2,v3): # (v1 cross v2) dot v3
	v1 = v1/LA.norm(v1)  # make v1,v2,v3 all with length 1. 
	v2 = v2/LA.norm(v2)
	v3 = v3/LA.norm(v3)
	temp = np.cross(v1,v2)
	# print(temp)
	return np.sum(temp*v3)

def calculat_dihedral(x): # X is 5 by 3 array, 5 points, 3 dimension x,y,z
	v1 = x[1] - x[0]
	v2 = x[2] - x[1]
	v3 = x[3] - x[2]
	v4 = x[4] - x[3]
	nv1 = np.cross(v1,v2)
	nv2 = np.cross(v2,v3)
	nv3 = np.cross(v3,v4)
	n1 = LA.norm(nv1)
	n2 = LA.norm(nv2)
	n3 = LA.norm(nv3)
	chi1 = compute_chirality(v1,v2,v3)
	chi2 = compute_chirality(v2,v3,v4)

	dih1 = np.arccos(np.dot(nv1,nv2)/n1/n2)*np.sign(chi1) # dihedral have +/-, depend on chirality
	dih2 = np.arccos(np.dot(nv2,nv3)/n2/n3)*np.sign(chi2)
	return dih1, dih2


# compute Rg, chirality, h2t all together
def compute_Rg(coord):
	# check and repair PBC breaking chain
	for i in range(NC-1): # start from the first C atom, loop over NC-1 bonds
		## PBC treatment
		delta = coord[i]-coord[i+1]
		select = delta>L/2 # find index either x,y,z that has difference larger than L/2, it means the 2nd atom is too large, so its image is too low
		coord[i+1,select] = coord[i+1,select]+L 
		select = delta<-L/2 #this means the 2nd atom is too low, so its image is too high
		coord[i+1,select] = coord[i+1,select]-L 

	# centeringw
	C = np.sum(coord,axis=0)/NC # center of mass
	coord = coord - C # center to 0.

	# get gyration tensor
	T = np.zeros((3,3))
	for i in range(3):
		for j in range(3):
			T[i,j] = np.sum(coord[:,i]*coord[:,j])/NC # for index i,j sum over all atoms

	# eigen desomposition
	w, v = LA.eig(T)
	order = np.argsort(w)  # sort eigenvalue and eigenvectors as descending order
	order = order[::-1] # from large to small
	w=w[order] # sort eigen values

	 # gyration components
	Rg = np.sqrt(np.sum(w))

	w=np.sqrt(w)
	
	# compute chirality
	v1 = coord[chiral_atom1]-coord[0]
	v2 = coord[chiral_atom2]-coord[chiral_atom1]
	v3 = coord[NC-1]-coord[chiral_atom2]
	# print(v1,v2,v3)
	Chi = compute_chirality(v1,v2,v3)

	# h2t = dist(coord[0],coord[NC-1]) # coord is modified and is complet, no need to consider PBC
	# head and tail may be at the edge of boundaies.
	h2t = np.sqrt(np.sum((coord[0]-coord[NC-1])**2)) 

	if NC==5:
		dih1,dih2 = calculat_dihedral(coord)
	else:
		dih1 = 0 
		dih2 = 0

	return Rg, h2t,Chi, w, dih1, dih2

# make bar plot
def std_hist_bar(index,data,width,xlabel,ylabel,title,filename):
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(index,data,width)
	ax.set_xlabel(xlabel,fontsize=20)
	ax.set_ylabel(ylabel,fontsize=20)
	plt.title(title)
	plt.savefig(filename,dpi=300)

# make curve plot
def std_hist_distribution(index,data,xlabel,ylabel,title,filename):
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.plot(index,data,linewidth=2,)
	ax.set_xlabel(xlabel,fontsize=20)
	ax.set_ylabel(ylabel,fontsize=20)
	plt.title(title)
	plt.savefig(filename,dpi=300)

# make scatter plot
def std_scatter(x,y,color,xlabel,ylabel,size,title,filename):
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.scatter(x,y, c=color,marker='o',s=size,linewidths=0, cmap='jet') # marker=','is single pixel, 
	ax.set_xlabel(xlabel,fontsize=30)
	ax.set_ylabel(ylabel,fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	
	if not isinstance(color,str):
		plt.colorbar()
	plt.title(title)
	plt.savefig(filename,dpi=300)

# obtain breaking time prob. dens. dist. but may not perfect exponential, there is equilibrium time
def find_best_startime(calc,deltat,maxt,scale):
	ErrorMIN = float('inf') # initial Error
	bestT = 0 # temperal best start time
	times = np.arange(0,Maxstartime,deltat) # searching startimes
	Errors = np.zeros(len(times)) # array for storing correspond errors
	for i in range(len(times)): # loop over test times
		time = times[i] 
		selectT = (calc[:,0]/scale>time) # select all breaking longer than minimum threshhold
		bins = np.arange(0,maxt/scale+1-time,dt) # modify and shift the range
		count, bins = np.histogram(calc[selectT,0]/scale-time,bins) # rescale and shift all breaking time
		avg_lifetime = np.mean(calc[selectT,0])/scale-time #calculate mean breaking time
		# print('lifetime =', avg_lifetime)

		K = 1/avg_lifetime  # all breaking rate
		fit = K*np.exp(-K*bins) # exponential distribution fitting

		prob_density = count/np.sum(count)/dt # measured prob. density
		Error = np.sqrt(np.sum((prob_density-fit[:-1])**2)/len(prob_density)) # len(bin) is change, need to take average
		# print(Error)
		Errors[i] = Error

		if Error<ErrorMIN:
			ErrorMIN = Error 
			bestT = time
	# print(Errors)

	# plot error vs startime
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(times,Errors, c = 'blue', linewidth=2) 
	ax.set_xlabel('offset time (ps)',fontsize=30)
	ax.set_ylabel('Error',fontsize=30)
	plt.title('optimum offset = '+str('{:-4.3f}'.format(bestT))+' ps',fontsize=25)
	plt.axvline(x=bestT, color='red', linestyle = '--',linewidth = 2)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('finding_startime.pdf',dpi=300)	

	return bestT

# combine all interations in Ni simulations, make .npy file to quickly load
def Combine_traj(i):
	if i%100==0:
		print(i)

	if NC==5:
		initialk = 0 #skip first 10 ps
	elif NC==12:
		initialk = 0
	elif NC==13:
		initialk = 0

	k=initialk # counter of iterations, C5 k=1, C12 k=0

	while 1:
		dumpname = path+str(i+1)+'/dump'+str(k)+'.reax.lammpstrj' # dump file name
		speciesname = path+str(i+1)+'/species'+str(k)+'.ALKO2'  # species file name
		if os.path.exists(dumpname):  # if there is still iterations
			if k==initialk:  # if it's the first iteration
				trajectory=load_dump(dumpname,Nf,NC,head) # get the trajectory of this iteration
				file = open(speciesname,'r')
				alltext = file.read()  # get species of this iteration
			else:
				coord = load_dump(dumpname,Nf,NC,head) # get the trajectory of this iteration
				trajectory=np.concatenate((trajectory,coord)) # combine trajectory of all iterations
				file = open(speciesname,'r')
				text = file.read()  # get species of this iteration
				alltext = alltext+text  # combine species of all iterations

			k=k+1 # next iteration
		else:
			break # if no file for the next iteration, stop
	# print(trajectory.shape)
	np.save('dumps/dump'+str(i+1)+'.npy',trajectory) #save combined trajectories

	file = open('species/species'+str(i+1)+'.txt','w')
	file.write(alltext)  # save conbined species.
	file.close()

#
#
#
#
#
#
#
#
#
#
#
############################
# Common part
############################
#
#
#
#
#
#
#
#
#
#
#

# combine all trajectory of C and species

print('Combine =',Combine)  # if there is need to combine the data
if Combine:
	print('start combining')
	os.system("rm -r dumps species")
	os.system("mkdir dumps species")

	inputs = np.arange(0,Ni,1) # use paralle to combine iterations
	result = Parallel(n_jobs=8)(delayed(Combine_traj)(I) for I in inputs)

if calc_flag == 1:

	COORD_b = np.zeros((Ni,NC,3))
	print('start finding breaking time')
	calc = np.zeros((Ni,10)) #time, bondIndex, Rg,xi1,xi2,xi3 h2t, chiratlity of breaking,dih1,dih2

	# use parallel computing to find breaking bond and time
	inputs = np.arange(0,Ni,1)
	result = Parallel(n_jobs=8)(delayed(find_breaking_bond_array)(I) for I in inputs)

	# print(result.shape)
	print('start computing breaking features')
	for i in np.arange(0,Ni,1):
		if i%100==0:
			print('i=',i)



	# determine breaking time, and breaking location,1-NC, check Cn, rather than number of species. because there can be H that's dissaociate, rather than
	# the breaking of C-C bond. 
		# speciesname = 'species/species'+str(i+1)+'.txt'
		# dumpname = 'dumps/dump'+str(i+1)+'.npy'

		# bond_len,coord,bindex,t = find_breaking_bond(dumpname)

		# if i>1800:
		# 	print(i,result[i])

		coord,bindex,t = result[i]

		COORD_b[i] = coord
		for j in range(NC-1): # start from the first C atom, loop over NC-1 bonds
			## PBC treatment
			delta = COORD_b[i,j]-COORD_b[i,j+1]
			select = delta>L/2 # find index either x,y,z that has difference larger than L/2, it means the 2nd atom is too large, so its image is too low
			COORD_b[i,j+1,select] = COORD_b[i,j+1,select]+L 
			select = delta<-L/2 #this means the 2nd atom is too low, so its image is too high
			COORD_b[i,j+1,select] = COORD_b[i,j+1,select]-L
		# t1 = find_breaking_time(speciesname)

		# print('t and t1 =',t, t1,'error is ', abs(t-t1)/t*100,'%')

		Rg, h2t, Chi, w,dih1,dih2 = compute_Rg(coord) 

		calc[i,0] = t
		calc[i,1] = bindex
		calc[i,2] = Rg
		calc[i,3] = w[0]
		calc[i,4] = w[1]
		calc[i,5] = w[2]
		calc[i,6] = h2t
		calc[i,7] = Chi
		calc[i,8] = dih1
		calc[i,9] = dih2

	# print(calc)
	np.save('calc.npy',calc)
	np.save('COORD_b.npy',COORD_b)
else:
	calc = np.load('calc.npy')
	COORD_b = np.load('COORD_b.npy')

calc_full = calc.copy() # select simulation IDs
calc = calc[sim0:sim1,:]
COORD_b = COORD_b[sim0:sim1]

########
# breaking time distribution, compute rate K
########

maxt = np.max(calc[:,0]) # maximum breaking time
print('max breaking time =',maxt/scale,'ps')

maxRg = np.max(calc[:,2]) # maximum Rg
print('max Rg =',maxRg)

maxh2t = np.max(calc[:,6]) # maximum h2t
print('max bh2t =',maxh2t)

maxw1 = np.max(calc[:,3]) # maximum h2t
print('max w1 =',maxw1)

startime = find_best_startime(calc,deltat,maxt,scale)
print('best start time = ',startime,' ps')

selectT = (calc[:,0]/scale>startime) # select all breaking longer than minimum threshhold
print('valid points number:',np.sum(selectT))

bins = np.arange(0,maxt/scale+1-startime,dt) # modify and shift the range
count, bins = np.histogram(calc[selectT,0]/scale-startime,bins) # rescale and shift all breaking time
avg_lifetime = np.mean(calc[selectT,0])/scale-startime #calculate mean breaking time
print('avg_lifetime =', avg_lifetime)

K = 1/avg_lifetime  # all breaking rate
fit = K*np.exp(-K*bins) # exponential distribution fitting

####
# plot breaking time and fitting
####
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
plt.plot(bins[:-1],count/np.sum(count)/dt, c = 'blue', linewidth=2,label='measurement') 
plt.plot(bins,fit, c = 'red', linewidth=2,label='fitting') 
plt.axvline(x=avg_lifetime, color='red', linestyle = '--',linewidth = 2)
plt.legend(loc='upper right', fontsize = 20)
ax.text(avg_lifetime+2,np.max(fit)/2, 'average lifetime = '+str('{:.3f}'.format(avg_lifetime))+'ps \n'+ r'$\lambda = $'+str('{:.3f}'.format(1/avg_lifetime))+r'ps$^{-1}$', fontsize=25)
ax.set_xlabel('lifetime (ps)',fontsize=30)
ax.set_ylabel(r'probability density (ps$^{-1}$)',fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('break_t.pdf',dpi=300)

###
# plot breaking time full distribution
###
bins = np.arange(0,maxt/scale+1,dt)
count, bins = np.histogram(calc[:,0]/scale,bins)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
prob_density_temp = count/np.sum(count)/dt
plt.plot(bins[:-1],prob_density_temp, c = 'blue', linewidth=2) 
plt.axvline(x=startime, color='red', linestyle = '--',linewidth = 2)
plt.text(startime+startime,np.max(prob_density_temp)*0.6,'offset cut = '+str('{:-4.3f}'.format(startime))+' ps',fontsize=25)
ax.set_xlabel('lifetime (ps)',fontsize=30)
ax.set_ylabel(r'probability density (ps$^{-1}$)',fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('break_t_full.pdf',dpi=300)


####
# breaking all bond probabitlity bar
####

# print(calc[:,1])
select = (calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime)
count, bins = np.histogram(calc[select,1],bins=np.arange(-0.5,NC-0.5,1))
# print(calc[:,1],bins)
prob = count/np.sum(count)
print(prob)
# print(np.sum(prob))
# std_hist_bar(bins[:-1]+1.5,count,1,'bond ID','counts','bond breaking distribution','breakingbond.pdf')


fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
rects1 = plt.bar(bins[:-1]+1.5,prob,0.9)
ax.set_xlabel('bond ID',fontsize=30)
ax.set_ylabel('probability',fontsize=30)
# plt.title(title)
plt.xticks(np.arange(1,NC,1), np.arange(1,NC,1))
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('bond_breaking_prob_full.pdf',dpi=300)

avg_prob = (prob+np.flip(prob))/2
# print('prob:',prob)
# print('avg_prob:',avg_prob)

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
rects1 = plt.bar(bins[:int(NC/2)]+1.5,avg_prob[:int(NC/2)],0.9)
ax.set_xlabel('bond ID',fontsize=30)
ax.set_ylabel('probability',fontsize=30)
# plt.title(title)
plt.xticks(np.arange(1,int(NC/2+1),1), np.arange(1,int(NC/2+1),1))
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('bond_breaking_prob_half.pdf',dpi=300)


######
#### find error bars of the integrated bond probability
######

chunK = 5
bond_valid = calc[select,1]
np.random.shuffle(bond_valid)  # shuffle all point
N_valid = len(bond_valid)      # randomly dvide into 5 chunks
print('N_valid:',N_valid)
n_valid = int(N_valid/chunK)
bins = np.arange(-0.5,NC-0.5,1)
COUNT = np.zeros((chunK,len(bins)-1))


for i in range(chunK):
	start = n_valid*i 
	end = start + n_valid 
	count, bins = np.histogram(bond_valid[start:end],bins)
	COUNT[i,:] = count
	# if i==0:
	# 	COUNT = count
	# else:
	# 	COUNT = np.concatenate((COUNT,count),axis=0)

print(COUNT.shape)


PROB = COUNT/np.sum(COUNT,axis=1,keepdims=True)
mean_prob = np.mean(PROB,axis = 0)
print(mean_prob)
stderror = np.std(PROB,axis=0)/np.sqrt(chunK)
# print(PROB)
print('stderror:',stderror)


#### plot full bond probability with error bar
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
rects1 = plt.bar(bins[:-1]+1.5,mean_prob,0.9)
plt.errorbar(bins[:-1]+1.5, mean_prob, stderror, fmt='o', color='red')
ax.set_xlabel('bond ID',fontsize=30)
ax.set_ylabel('probability',fontsize=30)
# plt.title(title)
plt.xticks(np.arange(1,NC,1), np.arange(1,NC,1))
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('bond_breaking_prob_full_ErrorBar.pdf',dpi=300)



avg_PROB = (PROB+np.flip(PROB,axis=1))
mean_avg_PROB = np.mean(avg_PROB,axis = 0)
stderror = np.std(avg_PROB,axis=0)/np.sqrt(chunK)

def pro2rate(p):
	return p*K

def rate2pro(r):
	return r/K

###### plot half bond probability with error bar
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
rects1 = plt.bar(bins[:int(NC/2)]+1.5,mean_avg_PROB[:int(NC/2)],0.9)
plt.errorbar(bins[:int(NC/2)]+1.5, mean_avg_PROB[:int(NC/2)], stderror[:int(NC/2)], fmt='o', color='red')
secax = ax.secondary_yaxis('right',functions=(pro2rate,rate2pro))
secax.set_ylabel(r'breaking rate (ps$^{-1}$)',fontsize=30)
secax.tick_params(labelsize=18)
ax.set_xlabel('bond ID',fontsize=30)
ax.set_ylabel('probability',fontsize=30)
# plt.title(title)
plt.xticks(np.arange(1,int(NC/2+1),1), np.arange(1,int(NC/2+1),1))
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('bond_breaking_prob_half_ErrorBar.pdf',dpi=300)


#### plot combined bond probability with error bar
if NC==13:
	avg_PROB = (PROB+np.flip(PROB,axis=1))
	avg_PROB_CG = np.zeros((chunK,3))
	avg_PROB_CG[:,[0]] = avg_PROB[:,[0]]+avg_PROB[:,[1]]
	avg_PROB_CG[:,[1]] = avg_PROB[:,[2]]+avg_PROB[:,[3]]
	avg_PROB_CG[:,[2]] = avg_PROB[:,[4]]+avg_PROB[:,[5]]
	mean_avg_PROB_CG = np.mean(avg_PROB_CG,axis = 0)
	stderror = np.std(avg_PROB_CG,axis=0)/np.sqrt(chunK)

	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(bins[0:3]+1.5,mean_avg_PROB_CG,0.9)
	plt.errorbar(bins[0:3]+1.5, mean_avg_PROB_CG, stderror, fmt='o', color='red')
	secax = ax.secondary_yaxis('right',functions=(pro2rate,rate2pro))
	secax.set_ylabel(r'breaking rate (ps$^{-1}$)',fontsize=30)
	secax.tick_params(labelsize=18)
	ax.set_xlabel('bond ID',fontsize=30)
	ax.set_ylabel('probability',fontsize=30)
	# plt.title(title)
	plt.xticks(np.array([1,2,3]), ['I','II','III'])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('bond_breaking_prob_CG_ErrorBar.pdf',dpi=300)



###
# sample original equilibrium distribution
###
if resample:
	COORD=0 # set COORD initially to be int	
	for i in np.arange(sim0,sim1,1):
		dumpname = 'dumps/dump'+str(i+1)+'.npy'
		coord = np.load(dumpname) 
		
		# COORD_break[i] = coord[int(calc[i,0])] # breaking configuration not useful

		# print(int(calc[i,0]-1))
		if (calc_full[i,0]-scale*startime)>1: 
			samplelimit = calc_full[i,0]-startime*scale # modify sampling upper limit
			choice = np.random.choice(int(samplelimit), ndmap, replace=True) # randomly select ndmap frames from each iteration
			choice = choice+startime*scale # make sure just sample the frames between startt ime and braking time
		else: # if by accident, it's too early breaking
			# choice = np.arange(0,ndmap,1)
			continue

		# choice = np.append(choice,calc[i,0]) # esnure the last frame is breaking frame.
		choice = choice.astype(int)
		
		coord_choice = coord[choice]

		if isinstance(COORD,int):  #i==0: # if COORD is still int, it means this is the first time we find valid coord_choice, not jsut i=0
			COORD = coord_choice
		else:
			COORD = np.concatenate((COORD,coord_choice)) # add to sampled list

### PBC treatment of sampled COORD

	print('start computing equilibrium features')

	for i in range(len(COORD)):
		for j in range(NC-1): # start from the first C atom, loop over NC-1 bonds
			## PBC treatment
			delta = COORD[i,j]-COORD[i,j+1]
			select = delta>L/2 # find index either x,y,z that has difference larger than L/2, it means the 2nd atom is too large, so its image is too low
			COORD[i,j+1,select] = COORD[i,j+1,select]+L 
			select = delta<-L/2 #this means the 2nd atom is too low, so its image is too high
			COORD[i,j+1,select] = COORD[i,j+1,select]-L

####
# calculate feature for original distribution sampling
###
	calc_choice = np.zeros((len(COORD),8))  # features for original sampling
	for i in range(len(COORD)):
		if i%1000==0:
			print(i)

		Rg, h2t, Chi, w, dih1,dih2 = compute_Rg(COORD[i]) 
		calc_choice[i,0] = h2t
		calc_choice[i,1] = Chi
		calc_choice[i,2] = Rg
		calc_choice[i,3] = w[0]
		calc_choice[i,4] = w[1]
		calc_choice[i,5] = w[2]
		calc_choice[i,6] = dih1
		calc_choice[i,7] = dih2

	np.save('COORD.npy',COORD)
	np.save('calc_choice.npy',calc_choice)

else:
	COORD = np.load('COORD.npy')
	calc_choice = np.load('calc_choice.npy')

print('equilibirum sampling len from valid simulations:',len(COORD))


#####
# equilibirum and breaking Rg probability density distribution
#####

# binN = 30
# dRg = 0.4
maxRg = np.max(calc_choice[:,2])
minRg = np.min(calc_choice[:,2])
print('min max Rg =',minRg,maxRg)

bins = np.arange(minRg-0.05,maxRg+0.05,dRg)
countRg_original, binsRg_original = np.histogram(calc_choice[:,2],bins)

# breaking Rg distribution
select = (calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime)
# print(np.sum(select))
countRg_br, binsRg_br = np.histogram(calc[select,2],bins)

# print(np.sum(countRg_br))

# probability density
rhoRg_original = countRg_original/np.sum(countRg_original)/dRg
rhoRg_breaking = countRg_br/np.sum(countRg_br)/dRg

rhoRg_original[countRg_original<20] = float('nan')
rhoRg_breaking[countRg_br<10] = float('nan')

print('countRG_original:',countRg_original)
print('countRg_br:',countRg_br)


print('rhoRg_original:',rhoRg_original)
print('rhoRg_breaking:',rhoRg_breaking)

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
plt.plot(bins[:-1],rhoRg_original, c = 'blue', marker='s',linewidth=2,label=r'$\rho_e(R_g)$') 
plt.plot(bins[:-1],rhoRg_breaking, c='r',marker='s',linewidth=2,label=r'$\rho_b(R_g)$') 
# plt.axvline(x=avg_lifetime, color='red', linestyle = '--',linewidth = 2)
plt.legend(loc='upper left', fontsize = 20)
# ax.text(20,0.05, 'average lifetime = '+str('{:.3f}'.format(avg_lifetime))+'ps \n'+ r'$\lambda = $'+str('{:.3f}'.format(1/avg_lifetime))+r'ps$^{-1}$', fontsize=20)
ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
ax.set_ylabel(r'probability density ($\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('Rg distribution.pdf',dpi=300)

# exit()
#
#
#
#
#
#
#
#
#
#
#
############################
# special part C5
############################
#
#
#
#
#
#
#
#
#
#
#

if NC==5: ## for C5H12


	########
	# breaking rate, and bond breaking rate
	########

	### braking rate as functino of Rg
	rate_breaking = rhoRg_breaking/rhoRg_original*K

	### calculate I bond probability and braking rate as a functino of Rg
	select_I = ((calc[:,1]==0)|(calc[:,1]==3))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime)
	print('I:',np.sum(select_I))
	countRgI, binsRg = np.histogram(calc[select_I,2],bins)
	countRgI = countRgI.astype('float')
	prob_RgI = countRgI/countRg_br   # probability
	rate_breaking_I = rate_breaking*prob_RgI  #braking rate vs Rg
	select_temp = rate_breaking==0
	rate_breaking_I[select_temp] = 0

	### calculate 03 bond probability and braking rate as a functino of Rg
	select_II = ((calc[:,1]==1)|(calc[:,1]==2))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('II:',np.sum(select_II))
	countRgII, binsRg = np.histogram(calc[select_II,2],bins)
	countRgII = countRgII.astype('float')
	prob_RgII = countRgII/countRg_br  # probability
	rate_breaking_II = rate_breaking*prob_RgII  #braking rate vs Rg
	rate_breaking_II[select_temp] = 0


	select_NaNbond = (countRgI<2)|(countRgII<2)
	countRgI[select_NaNbond] = float('nan')
	countRgII[select_NaNbond] = float('nan')
	# countRgIII[select_NaNbond] = float('nan')

	prob_RgI[select_NaNbond] = float('nan')	
	prob_RgII[select_NaNbond] = float('nan')	
	# prob_RgIII[select_NaNbond] = float('nan')	

	rate_breaking_I[select_NaNbond] = float('nan')	
	rate_breaking_II[select_NaNbond] = float('nan')	
	# rate_breaking_III[select_NaNbond] = float('nan')
	rate_breaking[select_NaNbond] = float('nan')	

	### plot I,II breaking probability vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(binsRg_original[:-1],prob_RgI, marker='s',markersize = 10,c = 'blue', label = 'bond I',linewidth=2) 
	plt.plot(binsRg_original[:-1],prob_RgII, marker='s',markersize = 10,c = 'red', label = 'bond II',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('probability',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left', fontsize = 20)
	plt.savefig('bond_probability.pdf',dpi=300)

	# plot I,II bond count density
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(binsRg_original[:-1],countRgI/dRg,marker='s',markersize = 10, c = 'blue',linestyle='-', label = 'bond I',linewidth=2) 
	plt.plot(binsRg_original[:-1],countRgII/dRg,marker='s',markersize = 10, c = 'red', label = 'bond II',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('counts density ($\mathrm{\mathring{A}}^{-1}$)',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left', fontsize = 20)
	plt.savefig('bond_count.pdf',dpi=300)


	## plot all/I/II breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rate_breaking, c = 'black', marker='s',markersize=5,label = r'$\lambda_{\mathrm{I+II}}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I, c = 'blue', marker='s',markersize=5,label = r'$\lambda_{\mathrm{I}}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II, c = 'red', marker='s',markersize=5,label = r'$\lambda_{\mathrm{II}}$',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'rate density (ps$^{-1}\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left',ncol=3, fontsize = 20)
	plt.savefig('breaking_rate.pdf',dpi=300)

	## zoomed in plot all/I/II breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rate_breaking, c = 'black', marker='s',markersize=12,label = r'$\lambda$',linewidth=5) 
	plt.plot(bins[:-1],rate_breaking_I, c = 'blue', marker='s',markersize=12,label = r'$\lambda_{I}$',linewidth=5) 
	plt.plot(bins[:-1],rate_breaking_II, c = 'red', marker='s',markersize=12,label = r'$\lambda_{II}$',linewidth=5) 	
	ax.set_xlabel('Rg',fontsize=30)
	ax.set_ylabel('breaking rate',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.xlim((1.45,1.75))
	plt.ylim((0,0.06))
	# plt.legend(loc='upper right', fontsize = 20)
	plt.savefig('breaking_rate_zoomin.pdf',dpi=300)

	####
	# integrated bond breaking rate
	####

	#method 1: using all bond breaking counts

	print('bond prob:',prob,'prob sum:',np.sum(prob))
	KI = K*(prob[1]+prob[2])
	KII = K*(prob[0]+prob[3])
	print(KI,KII)

	#method 2: using full definition 

	tempI = K*rhoRg_breaking*prob_RgI*dRg
	select_temp = np.isfinite(tempI)
	KI=np.sum(tempI[select_temp])
	tempII = K*rhoRg_breaking*prob_RgII*dRg
	select_temp = np.isfinite(tempII)
	KII=np.sum(tempII[select_temp])
	print(KI,KII)

	### plot integrate I,II breaking rate comparison.
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(np.array([1,2]),np.array([KI,KII]),0.9)
	ax.set_xlabel('bond ID',fontsize=30)
	ax.set_ylabel(r'breaking rate (ps$^{-1}$)',fontsize=30)
	# plt.title(title)
	plt.xticks(np.array([1,2]), ['I','II'])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('bond_breaking_rate.pdf',dpi=300)

	### plot I,II breaking probability comparison.
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(np.array([1,2]),np.array([KI,KII])/K,0.9)
	ax.set_xlabel('bond ID',fontsize=30)
	ax.set_ylabel('probability',fontsize=30)
	# plt.title(title)
	plt.xticks(np.array([1,2]), ['I','II'])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('bond_breaking_prob_group.pdf',dpi=300)

#
#
#
#
#
#
#
#
#
#
#
############################
# special part C13
############################
#
#
#
#
#
#
#
#
#
#
#

elif NC==13:

	### braking rate as functino of Rg
	rate_breaking = rhoRg_breaking/rhoRg_original*K

	select_I = ((calc[:,1]==0)|(calc[:,1]==1)|(calc[:,1]==10)|(calc[:,1]==11))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('I:',np.sum(select_I))
	countRgI, binsRg = np.histogram(calc[select_I,2],bins)
	countRgI = countRgI.astype('float')
	prob_RgI = countRgI/countRg_br  # probability
	rate_breaking_I = rate_breaking*prob_RgI  #braking rate vs Rg
	select_temp = rate_breaking==0
	rate_breaking_I[select_temp] = 0

	### calculate II bond probability and braking rate as a functino of Rg
	select_II = ((calc[:,1]==2)|(calc[:,1]==3)|(calc[:,1]==8)|(calc[:,1]==9))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('II:',np.sum(select_II))
	countRgII, binsRg = np.histogram(calc[select_II,2],bins)
	countRgII = countRgII.astype('float')
	prob_RgII = countRgII/countRg_br  # probability
	rate_breaking_II = rate_breaking*prob_RgII  #braking rate vs Rg
	# select_temp = rate_breaking==0
	rate_breaking_II[select_temp] = 0

	### calculate III bond probability and braking rate as a functino of Rg
	select_III = ((calc[:,1]==4)|(calc[:,1]==5)|(calc[:,1]==6)|(calc[:,1]==7))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('III:',np.sum(select_III))
	countRgIII, binsRg = np.histogram(calc[select_III,2],bins)
	countRgIII = countRgIII.astype('float')
	prob_RgIII = countRgIII/countRg_br  # probability
	rate_breaking_III = rate_breaking*prob_RgIII  #braking rate vs Rg
	# select_temp = rate_breaking==0
	rate_breaking_III[select_temp] = 0

	### plot I,II,III breaking probability vs Rg

	select_NaNbond = (countRgI<2)|(countRgII<2)|(countRgIII<2)
	countRgI[select_NaNbond] = float('nan')
	countRgII[select_NaNbond] = float('nan')
	countRgIII[select_NaNbond] = float('nan')

	prob_RgI[select_NaNbond] = float('nan')	
	prob_RgII[select_NaNbond] = float('nan')	
	prob_RgIII[select_NaNbond] = float('nan')	

	rate_breaking_I[select_NaNbond] = float('nan')	
	rate_breaking_II[select_NaNbond] = float('nan')	
	rate_breaking_III[select_NaNbond] = float('nan')
	rate_breaking[select_NaNbond] = float('nan')	


	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],prob_RgI, c = 'red', marker='s',label = 'bond I',linewidth=2) 
	plt.plot(bins[:-1],prob_RgII, c = 'blue', marker='s',label = 'bond II',linewidth=2) 
	plt.plot(bins[:-1],prob_RgIII, c = 'green', marker='s',label = 'bond III',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('probability',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper center', fontsize = 20)
	plt.savefig('bond_probability.pdf',dpi=300)

	# plot I,II,III bond count density
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(binsRg[:-1],countRgI/dRg,markersize = 8, c = 'red',marker='s',linestyle='-', label = 'bond I',linewidth=2) 
	plt.plot(binsRg[:-1],countRgII/dRg,markersize = 8, c = 'blue', marker='s',label = 'bond II',linewidth=2) 
	plt.plot(binsRg[:-1],countRgIII/dRg,markersize = 8, c = 'green', marker='s',label = 'bond III',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('counts density ($\mathrm{\mathring{A}}^{-1}$)',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='lower center', fontsize = 20)
	plt.savefig('bond_count.pdf',dpi=300)


	## plot all/I/II/III breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rate_breaking, c = 'black', marker='s',markersize = 5,label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I, c = 'red', marker='s',markersize = 5,label = r'$\lambda_{\mathrm{I}}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II, c = 'blue', marker='s',markersize = 5,label = r'$\lambda_{\mathrm{II}}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_III, c = 'green', marker='s',markersize = 5,label = r'$\lambda_{\mathrm{III}}$',linewidth=2) 
	plt.ylim((0,0.8))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'rate density (ps$^{-1}\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left',ncol=4, fontsize = 18)
	plt.savefig('breaking_rate.pdf',dpi=300)


	## plot all/averaged I,II,III breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	# plt.plot(bins[:-1],rate_breaking, c = 'blue', label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I/4, c = 'red', marker='s',label = r'$\lambda_{\mathbf{I}}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II/4, c = 'blue', marker='s',label = r'$\lambda_{II}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_III/4, c = 'green', marker='s',label = r'$\lambda_{III}$',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'rate density (ps$^{-1}\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left',ncol=3, fontsize = 20)
	plt.savefig('avg_breaking_rate.pdf',dpi=300)


	## zoomed in plot all/03/12 breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rate_breaking, c = 'black', marker='s',markersize=17,label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I, c = 'red', marker='s',markersize=17,label = r'$\lambda_{I}$',linewidth=4) 
	plt.plot(bins[:-1],rate_breaking_II, c = 'blue', marker='s',markersize=17,label = r'$\lambda_{II}$',linewidth=4) 
	plt.plot(bins[:-1],rate_breaking_III, c = 'green', marker='s',markersize=17,label = r'$\lambda_{III}$',linewidth=4) 
	ax.set_xlabel('Rg',fontsize=30)
	ax.set_ylabel('breaking rate',fontsize=30)
	plt.xticks(fontsize = 26)
	plt.yticks(fontsize = 26)
	plt.xlim((2.6,4.0))
	plt.ylim((0.02,0.1))
	# plt.legend(loc='upper right', fontsize = 20)
	plt.savefig('breaking_rate_zoomin.pdf',dpi=300)

	####
	# integrated bond breaking rate
	####

	#method 1:

	print('bond prob:',prob,'prob sum:',np.sum(prob))
	KI = K*(prob[4]+prob[5]+prob[6]+prob[7])
	KII = K*(prob[2]+prob[3]+prob[8]+prob[9])
	KIII = K*(prob[0]+prob[1]+prob[10]+prob[11])
	print(KI,KII,KIII)

	#method 2:
	tempI = K*rhoRg_breaking*prob_RgI*dRg
	select_temp = np.isfinite(tempI)
	KI=np.sum(tempI[select_temp])

	tempII = K*rhoRg_breaking*prob_RgII*dRg
	select_temp = np.isfinite(tempII)
	KII=np.sum(tempII[select_temp])

	tempIII = K*rhoRg_breaking*prob_RgIII*dRg
	select_temp = np.isfinite(tempIII)
	KIII=np.sum(tempIII[select_temp])

	print(KI,KII,KIII)

	### plot integrate I II III breaking rate comparison.
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(np.array([1,2,3]),np.array([KI,KII,KIII]),0.9)
	ax.set_xlabel('bond ID',fontsize=30)
	ax.set_ylabel(r'breaking rate (ps$^{-1}$)',fontsize=30)
	# plt.title(title)
	plt.xticks(np.array([1,2,3]), [r'$\mathrm{I}$','II','III'])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('bond_breaking_rate.pdf',dpi=300)

	### plot average single bond integrate breaking rate comparison from I, II, III.
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(np.array([1,2,3]),np.array([KI/4,KII/4,KIII/4]),0.9)
	ax.set_xlabel('bond ID',fontsize=30)
	ax.set_ylabel(r'breaking rate (ps$^{-1}$)',fontsize=30)
	# plt.title(title)
	plt.xticks(np.array([1,2,3]), ['I','II','III'])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('avg_single_bond_breaking_rate.pdf',dpi=300)



	# exit()
########dmap for C13 





	# COORD = np.zeros((Ndmap,NC,3))

	# for i in range(Ni):
	# 	dumpname = 'dumps/dump'+str(i)+'.npy'
	# 	choice = np.random.choice(int(calc[i,0]-1), ndmap-1, replace=False)
	# 	choice = np.append(choice,calc[i,0]) # esnure the last frame is breaking frame.
	# 	choice = choice.astype(int)
	# 	coord = np.load(dumpname)
	# 	coord_choice = coord[choice]
	# 	if i==0:
	# 		COORD = coord_choice
	# 	else:
	# 		COORD = np.concatenate((COORD,coord_choice))

	# # PBC treatment of all configurations
	# for i in range(Ndmap):
	# 	for j in range(NC-1): # start from the first C atom, loop over NC-1 bonds
	# 		## PBC treatment for dmap
	# 		delta = COORD[i,j]-COORD[i,j+1]
	# 		select = delta>L/2 # find index either x,y,z that has difference larger than L/2, it means the 2nd atom is too large, so its image is too low
	# 		COORD[i,j+1,select] = COORD[i,j+1,select]+L 
	# 		select = delta<-L/2 #this means the 2nd atom is too low, so its image is too high
	# 		COORD[i,j+1,select] = COORD[i,j+1,select]-L


	Noriginal = len(COORD)
	print('Noriginal =',Noriginal)
	print(len(calc_choice))

	dmapselect = np.random.choice(Noriginal,1000,replace = False)
	nystromselect = np.random.choice(Noriginal,3000,replace = False)

	COORD_dmap = COORD[dmapselect]
	COORD_nystrom = COORD[nystromselect]
	# compute pdist
	pdist = dp.compute_pdist(COORD_dmap,1000,NC,h2tflag,mirrorflag)

	# conduct dmap
	evl,evc = dp.diffusion_maps(pdist,epsilon,exponent)

	# plot eigenvalues
	index = np.arange(1,11,1)  # select only first 10 eigenvalues to show
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	fig = plt.bar(index , evl[0:10], 0.9,
	                 alpha=0.9,
	                 color='dodgerblue',
	                 label='Cross-validation error ')
	plt.xticks(index, index)
	plt.axhline(y=0.35, color='red', linestyle='--', linewidth = 2) # add a horizontal dashed line
	ax.set_xlabel('Eigen Index',fontsize=30)
	ax.set_ylabel('Eigenvalues',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	# plt.ylim((0,0.15))
	plt.savefig('eigenvalue.pdf',dpi=300)

	# compute Gr,h2t,xi1,2,3,Chi for selected points
	# calc_choice = np.zeros((Ndmap,6))
	# for i in range(Ndmap):
	# 	Rg, h2t, Chi, w = compute_Rg(COORD[i]) 
	# 	calc_choice[i,0] = h2t
	# 	calc_choice[i,1] = Chi
	# 	calc_choice[i,2] = Rg
	# 	calc_choice[i,3] = w[0]
	# 	calc_choice[i,4] = w[1]
	# 	calc_choice[i,5] = w[2]
	# std_scatter(evc[:,1],calc_choice[dmapselect,2],'b',r'$\xi_{1}$',r'R$_g$ ($\mathrm{\mathring{A}}$)',20,'','evec1Rg.pdf')

	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.scatter(evc[:,1],calc_choice[dmapselect,2], c='b',marker='o',s=25,linewidths=0, cmap='jet') # marker=','is single pixel, 
	ax.set_xlabel(r'$\xi_{2}$',fontsize=30)
	ax.set_ylabel(r'R$_g$ ($\mathrm{\mathring{A}}$)',fontsize=30)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	# plt.title(title)
	plt.savefig('evec1Rg.pdf',dpi=300)

	print('corr evec1 Rg = ',np.corrcoef(evc[:,1],calc_choice[dmapselect,2])[0,1])

	exit()


	std_scatter(evc[:,1],evc[:,2],calc_choice[dmapselect,0],'evec 1','evec2',20,'evec12h2t','evec12h2t.pdf')
	std_scatter(evc[:,1],evc[:,2],calc_choice[dmapselect,1],'evec 1','evec2',20,'evec12Chi','evec12Chi.pdf')
	std_scatter(evc[:,1],evc[:,2],abs(calc_choice[dmapselect,1]),'evec 1','evec2',20,'evec12Chi','evec12absChi.pdf')
	std_scatter(evc[:,1],evc[:,2],calc_choice[dmapselect,2],'evec 1','evec2',20,'evec12Rg.','evec12Rg.pdf')
	std_scatter(evc[:,1],evc[:,2],calc_choice[dmapselect,3],'evec 1','evec2',20,'evec12w1.','evec12w1.pdf')
	std_scatter(evc[:,1],evc[:,2],calc_choice[dmapselect,4],'evec 1','evec2',20,'evec12w2.','evec12w2.pdf')
	std_scatter(evc[:,1],evc[:,2],calc_choice[dmapselect,5],'evec 1','evec2',20,'evec12w3.','evec12w3.pdf')

	std_scatter(evc[:,1],evc[:,3],calc_choice[dmapselect,0],'evec 1','evec3',20,'evec12h2t','evec13h2t.pdf')
	std_scatter(evc[:,1],evc[:,3],calc_choice[dmapselect,1],'evec 1','evec3',20,'evec12Chi','evec13Chi.pdf')
	std_scatter(evc[:,1],evc[:,3],abs(calc_choice[dmapselect,1]),'evec 1','evec3',20,'evec12Chi','evec13absChi.pdf')
	std_scatter(evc[:,1],evc[:,3],calc_choice[dmapselect,2],'evec 1','evec3',20,'evec12Rg.','evec13Rg.pdf')
	std_scatter(evc[:,1],evc[:,3],calc_choice[dmapselect,3],'evec 1','evec3',20,'evec12w1.','evec13w1.pdf')
	std_scatter(evc[:,1],evc[:,3],calc_choice[dmapselect,4],'evec 1','evec3',20,'evec12w2.','evec13w2.pdf')
	std_scatter(evc[:,1],evc[:,3],calc_choice[dmapselect,5],'evec 1','evec3',20,'evec12w3.','evec13w3.pdf')

	std_scatter(evc[:,1],calc_choice[dmapselect,0],'b','evec 1','h2t',20,'evec1h2t','evec1h2t.pdf')
	std_scatter(evc[:,1],calc_choice[dmapselect,2],'b','evec 1','Rg',20,'evec1Rg','evec1Rg.pdf')
	std_scatter(evc[:,1],calc_choice[dmapselect,3],'b','evec 1','w1',20,'evec1w1','evec1w1.pdf')
	std_scatter(calc_choice[dmapselect,0],calc_choice[dmapselect,2],'b','h2t','w1',20,'evec1w1','h2tRg.pdf')

	print('corr evec1 h2t = ',np.corrcoef(evc[:,1],calc_choice[dmapselect,0])[0,1])
	print('corr evec1 Rg = ',np.corrcoef(evc[:,1],calc_choice[dmapselect,2])[0,1])
	print('corr evec1 w1 = ',np.corrcoef(evc[:,1],calc_choice[dmapselect,3])[0,1])

	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	# select = (calc[:,0]/10000>0)&(calc[:,0]/10000<80) 
	Hb, xgrid, ygrid = np.histogram2d(evc[:,1],evc[:,2],bins = 5)
	Hb = Hb.transpose()
	Hb[Hb<5] = float('nan')
	Hbp = Hb/len(evc) 
	plt.imshow(Hbp,cmap='viridis',origin='lower')
	# plt.show()
	plt.colorbar()
	plt.savefig('breaking_prob_dmap2D.pdf',dpi=300)


	##### nystrom
	COORD_b =COORD_b[selectT]

	nyevc_b = dp.nystrom(COORD_b,COORD_dmap,evl,evc,NC, epsilon,exponent,h2tflag,mirrorflag)
	nyevc_e = dp.nystrom(COORD_nystrom,COORD_dmap,evl,evc,NC, epsilon,exponent,h2tflag,mirrorflag)

	print('nyevec_b size:',len(nyevc_b))
	print('nyevec_e size:',len(nyevc_e))

	# print(nyevc)
	std_scatter(nyevc_b[:,1],calc[selectT,2],'b','nyevec 1','Rg',20,'evec1h2t','nyevec1Rg_b.pdf')
	std_scatter(nyevc_e[:,1],calc_choice[nystromselect,6],'b','nyevec 1','h2t',20,'evec1h2t','nyevec1h2t_e.pdf')


	#####
	# equilibirum and breaking evec probability density distribution
	#####

	# binN = 30
	# dRg = 0.4
	maxevec = max(np.max(nyevc_e[:,1]),np.max(nyevc_b[:,1]))
	minevec = min(np.min(nyevc_e[:,1]),np.min(nyevc_b[:,1]))
	print('min max evec1 =',minevec,maxevec)

	bins = np.arange(minevec-0.01,maxevec+0.01,devec)
	countevec_e, binsevec_e = np.histogram(nyevc_e[:,1],bins)

	# breaking Rg distribution
	countevec_b, binsevec_b = np.histogram(nyevc_b[:,1],bins)

	# print(np.sum(countRg_br))

	# probability density
	rhoevec_e = countevec_e/np.sum(countevec_e)/devec
	rhoevec_b = countevec_b/np.sum(countevec_b)/devec

	rhoevec_e[countevec_e<2] = float('nan')
	rhoevec_b[countevec_b<2] = float('nan')

	print('countevec_e:',countevec_e)
	print('countevec_b:',countevec_b)


	print('rhoevec_e:',rhoevec_e)
	print('rhoevec_b:',rhoevec_b)

	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rhoevec_e, c = 'blue', linewidth=2,label=r'$\rho_e(evec)$') 
	plt.plot(bins[:-1],rhoevec_b, c='r',linewidth=2,label=r'$\rho_b(evec)$') 
	# plt.axvline(x=avg_lifetime, color='red', linestyle = '--',linewidth = 2)
	plt.legend(loc='upper right', fontsize = 20)
	# ax.text(20,0.05, 'average lifetime = '+str('{:.3f}'.format(avg_lifetime))+'ps \n'+ r'$\lambda = $'+str('{:.3f}'.format(1/avg_lifetime))+r'ps$^{-1}$', fontsize=20)
	ax.set_xlabel(r'evec ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'probability density ($\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('evec distribution.pdf',dpi=300)

	
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	# select = (calc[:,0]/10000>0)&(calc[:,0]/10000<80) 
	He, xgrid, ygrid = np.histogram2d(nyevc_e[:,1],nyevc_e[:,2],bins = 5)
	He = He.transpose()
	He[He<10] = float('nan')
	Hep = He/len(nyevc_e) 
	plt.imshow(Hep,cmap='viridis',origin='lower')
	# plt.show()
	plt.colorbar()
	plt.savefig('dmap2D_rho_e.pdf',dpi=300)


	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	# select = (calc[:,0]/10000>0)&(calc[:,0]/10000<80) 
	Hb, xgrid, ygrid = np.histogram2d(nyevc_b[:,1],nyevc_b[:,2],bins=(xgrid, ygrid))
	Hb = Hb.transpose()
	Hb[Hb<2] = float('nan')
	Hbp = Hb/len(nyevc_b) 
	plt.imshow(Hbp,cmap='viridis',origin='lower')
	# plt.show()
	plt.colorbar()
	plt.savefig('dmap2D_rho_b.pdf',dpi=300)

	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	Heb = Hbp/Hep
	plt.imshow(Heb,cmap='viridis',origin='lower')
	# plt.show()
	plt.colorbar()
	plt.savefig('dmap2D_rho_eb.pdf',dpi=300)
# exit()
	# location of the breaking configuration in dmap. Colored with breaking time, chirality, gr, h2t, location, etc. 
	# breaking_ID = np.arange(ndmap-1,Ndmap,ndmap) # final breaking frames from all initial simulations
	# fig = plt.figure(figsize=(8, 6))
	# ax = fig.gca()
	# plt.scatter(evc[:,1],evc[:,2], c='lightgrey',marker='o',s=40, alpha=0.6,linewidths=0) # marker=','is single pixel, 
	# plt.scatter(evc[breaking_ID,1],evc[breaking_ID,2], c='black',marker='o',s=50) # marker=','is single pixel, 
	# ax.set_xlabel('evec1',fontsize=20)
	# ax.set_ylabel('evec2',fontsize=20)
	# plt.title('breaking_points')
	# plt.savefig('breaking_points.pdf',dpi=300)


	rate_breaking = rhoevec_b/rhoevec_e*K

	select_I = ((calc[:,1]==4)|(calc[:,1]==5)|(calc[:,1]==6)|(calc[:,1]==7))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('I:',np.sum(select_I))
	breaking_bonds = calc[(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime),1]
	select_I = (breaking_bonds==4)|(breaking_bonds==5)|(breaking_bonds==6)|(breaking_bonds==7)
	countRgI, binsRg = np.histogram(nyevc_b[select_I,1],bins)
	countRgI = countRgI.astype('float')
	prob_RgI = countRgI/countevec_b  # probability
	rate_breaking_I = rate_breaking*prob_RgI  #braking rate vs Rg
	select_temp = rate_breaking==0
	rate_breaking_I[select_temp] = 0

	### calculate II bond probability and braking rate as a functino of Rg
	select_II = ((calc[:,1]==2)|(calc[:,1]==3)|(calc[:,1]==8)|(calc[:,1]==9))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('II:',np.sum(select_II))
	breaking_bonds = calc[(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime),1]
	select_II = (breaking_bonds==2)|(breaking_bonds==3)|(breaking_bonds==8)|(breaking_bonds==9)
	countRgII, binsRg = np.histogram(nyevc_b[select_II,1],bins)
	countRgII = countRgII.astype('float')
	prob_RgII = countRgII/countevec_b  # probability
	rate_breaking_II = rate_breaking*prob_RgII  #braking rate vs Rg
	# select_temp = rate_breaking==0
	rate_breaking_II[select_temp] = 0

	### calculate III bond probability and braking rate as a functino of Rg
	select_III = ((calc[:,1]==0)|(calc[:,1]==1)|(calc[:,1]==10)|(calc[:,1]==11))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('III:',np.sum(select_III))
	print('I:',np.sum(select_I))
	breaking_bonds = calc[(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime),1]
	select_III = (breaking_bonds==0)|(breaking_bonds==1)|(breaking_bonds==10)|(breaking_bonds==11)
	countRgIII, binsRg = np.histogram(nyevc_b[select_III,1],bins)
	countRgIII = countRgIII.astype('float')
	prob_RgIII = countRgIII/countevec_b  # probability
	rate_breaking_III = rate_breaking*prob_RgIII  #braking rate vs Rg
	# select_temp = rate_breaking==0
	rate_breaking_III[select_temp] = 0

	### plot I,II,III breaking probability vs Rg

	select_NaNbond = (countRgI<1)|(countRgII<1)|(countRgIII<1)
	countRgI[select_NaNbond] = float('nan')
	countRgII[select_NaNbond] = float('nan')
	countRgIII[select_NaNbond] = float('nan')

	prob_RgI[select_NaNbond] = float('nan')	
	prob_RgII[select_NaNbond] = float('nan')	
	prob_RgIII[select_NaNbond] = float('nan')	

	rate_breaking_I[select_NaNbond] = float('nan')	
	rate_breaking_II[select_NaNbond] = float('nan')	
	rate_breaking_III[select_NaNbond] = float('nan')	


	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],prob_RgI, c = 'red', label = 'bond-I',linewidth=2) 
	plt.plot(bins[:-1],prob_RgII, c = 'blue', label = 'bond-II',linewidth=2) 
	plt.plot(bins[:-1],prob_RgIII, c = 'green', label = 'bond-III',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('probability',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper center', fontsize = 20)
	plt.savefig('bond_probability_dmap.pdf',dpi=300)

	# plot I,II,III bond count density
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],countRgI/devec,markersize = 10, c = 'red',linestyle='-', label = 'bond-I',linewidth=2) 
	plt.plot(bins[:-1],countRgII/devec,markersize = 10, c = 'blue', label = 'bond-II',linewidth=2) 
	plt.plot(bins[:-1],countRgIII/devec,markersize = 10, c = 'green', label = 'bond-III',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('counts density ($\mathrm{\mathring{A}}^{-1}$)',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='lower center', fontsize = 20)
	plt.savefig('bond_count_dmap.pdf',dpi=300)


	## plot all/I/II/III breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rate_breaking, c = 'black', label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I, c = 'red', label = r'$\lambda_{I}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II, c = 'blue', label = r'$\lambda_{II}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_III, c = 'green', label = r'$\lambda_{III}$',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'rate density (ps$^{-1}\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left',ncol=1, fontsize = 20)
	plt.savefig('breaking_rate_dmap.pdf',dpi=300)


	## plot all/averaged I,II,III breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	# plt.plot(bins[:-1],rate_breaking, c = 'blue', label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I/4, c = 'red', label = r'$\lambda_{I}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II/4, c = 'blue', label = r'$\lambda_{II}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_III/4, c = 'green', label = r'$\lambda_{III}$',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'rate density (ps$^{-1}\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left',ncol=3, fontsize = 20)
	plt.savefig('avg_breaking_rate_dmap.pdf',dpi=300)



# exit()
#
#
#
#
#
#
#
#
#
#
#
############################
# special part C12
############################
#
#
#
#
#
#
#
#
#
#
#


elif NC==12:

	########
	# breaking rate, and bond breaking rate
	########

	### braking rate as functino of Rg
	rate_breaking = rhoRg_breaking/rhoRg_original*K

	### calculate I bond probability and braking rate as a functino of Rg
	select_I = ((calc[:,1]==4)|(calc[:,1]==5)|(calc[:,1]==6))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('I:',np.sum(select_I))
	countRgI, binsRg = np.histogram(calc[select_I,2],bins)
	prob_RgI = countRgI/countRg_br  # probability
	rate_breaking_I = rate_breaking*prob_RgI  #braking rate vs Rg
	select_temp = rate_breaking==0
	rate_breaking_I[select_temp] = 0

	### calculate II bond probability and braking rate as a functino of Rg
	select_II = ((calc[:,1]==2)|(calc[:,1]==3)|(calc[:,1]==7)|(calc[:,1]==8))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('II:',np.sum(select_II))
	countRgII, binsRg = np.histogram(calc[select_II,2],bins)
	prob_RgII = countRgII/countRg_br  # probability
	rate_breaking_II = rate_breaking*prob_RgII  #braking rate vs Rg
	# select_temp = rate_breaking==0
	rate_breaking_II[select_temp] = 0

	### calculate III bond probability and braking rate as a functino of Rg
	select_III = ((calc[:,1]==0)|(calc[:,1]==1)|(calc[:,1]==9)|(calc[:,1]==10))&(calc[:,0]/scale>startime)&(calc[:,0]/scale<endtime) # notice that there must be ():1|0&0=1, (1|0)&0=0 
	print('III:',np.sum(select_III))
	countRgIII, binsRg = np.histogram(calc[select_III,2],bins)
	prob_RgIII = countRgIII/countRg_br  # probability
	rate_breaking_III = rate_breaking*prob_RgIII  #braking rate vs Rg
	# select_temp = rate_breaking==0
	rate_breaking_III[select_temp] = 0

	### plot I,II,III breaking probability vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(binsRg_original[:-1],prob_RgI, c = 'red', label = 'bond-I',linewidth=2) 
	plt.plot(binsRg_original[:-1],prob_RgII, c = 'blue', label = 'bond-II',linewidth=2) 
	plt.plot(binsRg_original[:-1],prob_RgIII, c = 'green', label = 'bond-III',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('probability',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper center', fontsize = 20)
	plt.savefig('bond_probability.pdf',dpi=300)

	# plot I,II,III bond count density
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(binsRg_original[:-1],countRgI/dRg,markersize = 10, c = 'red',linestyle='-', label = 'bond-I',linewidth=2) 
	plt.plot(binsRg_original[:-1],countRgII/dRg,markersize = 10, c = 'blue', label = 'bond-II',linewidth=2) 
	plt.plot(binsRg_original[:-1],countRgIII/dRg,markersize = 10, c = 'green', label = 'bond-III',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel('counts density ($\mathrm{\mathring{A}}^{-1}$)',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='lower center', fontsize = 20)
	plt.savefig('bond_count.pdf',dpi=300)


	## plot all/I/II/III breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rate_breaking, c = 'black', label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I, c = 'red', label = r'$\lambda_{I}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II, c = 'blue', label = r'$\lambda_{II}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_III, c = 'green', label = r'$\lambda_{III}$',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'rate density (ps$^{-1}\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left',ncol=1, fontsize = 20)
	plt.savefig('breaking_rate.pdf',dpi=300)


	## plot all/averaged I,II,III breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	# plt.plot(bins[:-1],rate_breaking, c = 'blue', label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I/3, c = 'red', label = r'$\lambda_{I}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II/4, c = 'blue', label = r'$\lambda_{II}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_III/4, c = 'green', label = r'$\lambda_{III}$',linewidth=2) 
	# plt.ylim((0,0.1))
	ax.set_xlabel(r'Rg ($\mathrm{\mathring{A}}$)',fontsize=30)
	ax.set_ylabel(r'rate density (ps$^{-1}\mathrm{\mathring{A}}^{-1}$)',fontsize=25)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc='upper left',ncol=3, fontsize = 20)
	plt.savefig('avg_breaking_rate.pdf',dpi=300)


	## zoomed in plot all/03/12 breaking rate vs Rg
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	plt.plot(bins[:-1],rate_breaking, c = 'black', label = r'$\lambda$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_I, c = 'red', label = r'$\lambda_{I}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_II, c = 'blue', label = r'$\lambda_{II}$',linewidth=2) 
	plt.plot(bins[:-1],rate_breaking_III, c = 'green', label = r'$\lambda_{III}$',linewidth=2) 
	ax.set_xlabel('Rg',fontsize=30)
	ax.set_ylabel('breaking rate',fontsize=30)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.xlim((1.0,3.0))
	plt.ylim((0,0.1))
	# plt.legend(loc='upper right', fontsize = 20)
	plt.savefig('breaking_rate_zoomin.pdf',dpi=300)

	# print(rate_breaking)
	# print(rate_breaking_03)
	# print(rate_breaking_12)

	####
	# integrated bond breaking rate
	####

	#method 1:

	print('bond prob:',prob,'prob sum:',np.sum(prob))
	KI = K*(prob[4]+prob[5]+prob[6])
	KII = K*(prob[2]+prob[3]+prob[7]+prob[8])
	KIII = K*(prob[0]+prob[1]+prob[9]+prob[10])
	print(KI,KII,KIII)

	#method 2:
	tempI = K*rhoRg_breaking*prob_RgI*dRg
	select_temp = np.isfinite(tempI)
	KI=np.sum(tempI[select_temp])

	tempII = K*rhoRg_breaking*prob_RgII*dRg
	select_temp = np.isfinite(tempII)
	KII=np.sum(tempII[select_temp])

	tempIII = K*rhoRg_breaking*prob_RgIII*dRg
	select_temp = np.isfinite(tempIII)
	KIII=np.sum(tempIII[select_temp])

	print(KI,KII,KIII)

	### plot integrate I II III breaking rate comparison.
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(np.array([1,2,3]),np.array([KI,KII,KIII]),0.9)
	ax.set_xlabel('bond ID',fontsize=30)
	ax.set_ylabel(r'breaking rate (ps$^{-1}$)',fontsize=30)
	# plt.title(title)
	plt.xticks(np.array([1,2,3]), ['I','II','III'])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('bond_breaking_rate.pdf',dpi=300)

	### plot average single bond integrate breaking rate comparison from I, II, III.
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	rects1 = plt.bar(np.array([1,2,3]),np.array([KI/3,KII/4,KIII/4]),0.9)
	ax.set_xlabel('bond ID',fontsize=30)
	ax.set_ylabel(r'breaking rate (ps$^{-1}$)',fontsize=30)
	# plt.title(title)
	plt.xticks(np.array([1,2,3]), ['I','II','III'])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig('avg_single_bond_breaking_rate.pdf',dpi=300)







exit()
# dmap 

# sampele 5k configurations from 100 simulations, including the breaking configurations
COORD = np.zeros((Ndmap,NC,3))

for i in range(Ni):
	dumpname = 'dumps/dump'+str(i)+'.npy'
	choice = np.random.choice(int(calc[i,0]-1), ndmap-1, replace=False)
	choice = np.append(choice,calc[i,0]) # esnure the last frame is breaking frame.
	choice = choice.astype(int)
	coord = np.load(dumpname)
	coord_choice = coord[choice]
	if i==0:
		COORD = coord_choice
	else:
		COORD = np.concatenate((COORD,coord_choice))

# PBC treatment of all configurations
for i in range(Ndmap):
	for j in range(NC-1): # start from the first C atom, loop over NC-1 bonds
		## PBC treatment for dmap
		delta = COORD[i,j]-COORD[i,j+1]
		select = delta>L/2 # find index either x,y,z that has difference larger than L/2, it means the 2nd atom is too large, so its image is too low
		COORD[i,j+1,select] = COORD[i,j+1,select]+L 
		select = delta<-L/2 #this means the 2nd atom is too low, so its image is too high
		COORD[i,j+1,select] = COORD[i,j+1,select]-L


# compute pdist
pdist = dp.compute_pdist(COORD,Ndmap,NC)

# conduct dmap
evl,evc = dp.diffusion_maps(pdist)

# plot eigenvalues
index = np.arange(1,11,1)  # select only first 10 eigenvalues to show
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
fig = plt.bar(index , evl[0:10], 0.9,
                 alpha=0.9,
                 color='dodgerblue',
                 label='Cross-validation error ')
plt.xticks(index, index)
plt.axhline(y=0.03, color='red', linestyle='--', linewidth = 2) # add a horizontal dashed line
ax.set_xlabel('Eigen Index',fontsize=20)
ax.set_ylabel('Eigenvalues',fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
# plt.ylim((0,0.15))
plt.savefig('eigenvalue.pdf',dpi=300)

# compute Gr,h2t,xi1,2,3,Chi for selected points
calc_choice = np.zeros((Ndmap,6))
for i in range(Ndmap):
	Rg, h2t, Chi, w = compute_Rg(COORD[i]) 
	calc_choice[i,0] = h2t
	calc_choice[i,1] = Chi
	calc_choice[i,2] = Rg
	calc_choice[i,3] = w[0]
	calc_choice[i,4] = w[1]
	calc_choice[i,5] = w[2]


std_scatter(evc[:,1],evc[:,3],calc_choice[:,1],'evec 1','evec2','evec12Rg','evec12Rg.pdf')

# location of the breaking configuration in dmap. Colored with breaking time, chirality, gr, h2t, location, etc. 
breaking_ID = np.arange(ndmap-1,Ndmap,ndmap) # final breaking frames from all initial simulations
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
plt.scatter(evc[:,1],evc[:,2], c='lightgrey',marker='o',s=40, alpha=0.6,linewidths=0) # marker=','is single pixel, 
plt.scatter(evc[breaking_ID,1],evc[breaking_ID,2], c='black',marker='o',s=50) # marker=','is single pixel, 
ax.set_xlabel('evec1',fontsize=20)
ax.set_ylabel('evec2',fontsize=20)
plt.title('breaking_points')
plt.savefig('breaking_points.pdf',dpi=300)
