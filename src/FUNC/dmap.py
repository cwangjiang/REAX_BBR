# This function use 3D array trajectory, and Kabsch to return a N by N pdist matrix to do dmap

import numpy as np
from numpy import linalg as LA
import math
import FUNC.Kabsch as kbs
import matplotlib.pyplot as plt 



def compute_pdist(A,N,l,h2tflag,mirrorflag):  # A is the trajectory, N is number of frames, l is number of atoms being selected.

	Pd = np.zeros((N,N))

	print('Now start computing pairwise distance:')

	for i in range(N):
		if(i%100 == 0):
			print(i)
		for j in np.arange(i+1,N,1):
			# Pd[i,j] = kbs.Kabsch(A[i],A[j],l)
			Pd[i,j] = dist(A[i],A[j],l,h2tflag,mirrorflag)
			Pd[j,i] = Pd[i,j]

	return Pd

def dist(A,B,l,h2tflag,mirrorflag):  # A is the trajectory, N is number of frames, l is number of atoms being selected.

	d = kbs.Kabsch(A,B,l)

	if h2tflag:
		Bh2t = np.flip(B,axis=0)
		d = min(d,kbs.Kabsch(A,Bh2t,l))

	if mirrorflag:
		Bmirror = B*np.array([-1,1,1])
		d = min(d,kbs.Kabsch(A,Bmirror,l))
	return d

def diffusion_maps(Pdist,epsilon,exponent):
	LB_flag = 1

	#### Bandwidth epsilon ####

	# epsilon = epsilon #0.5

	D2 = np.power(Pdist,exponent)  # distance^2
	# print(D)
	# print(D2)

	del Pdist # delete Pdist to release memory

	K = np.exp(-D2/epsilon)  # kernel matrix
	# print(K)

	del D2 # delete D2 to release memory

	####  Normalization #####
	weighted = False

	if weighted == False:
		NormalMatrix = np.sum(K,axis = 1,keepdims=True) #Normalization matrix
	else:
		NormalMatrix = np.sum(K*Pivot_weight,axis = 1,keepdims=True) #Normalization matrix
	# print(N)
	NormalMatrix = np.diag(NormalMatrix[:,0])
	NormalMatrix_inv = LA.inv(NormalMatrix)
	# print(N)
	# print(N_inv)

	if LB_flag==1:  # FP normalization

		if weighted == False:
			Markov = np.matmul(NormalMatrix_inv,K)	# Markov matrix
		else:
			Markov = np.matmul(NormalMatrix_inv,K)*Pivot_weight	# Markov matrix

	else:		# LB normalization
		K = np.matmul(NormalMatrix_inv,K)
		# print(K)
		K = np.matmul(K,NormalMatrix_inv)
		# print(K)
		NormalMatrix = np.sum(K,axis = 1,keepdims=True)
		NormalMatrix = np.diag(NormalMatrix[:,0])
		NormalMatrix_inv = LA.inv(NormalMatrix)
		Markov = np.matmul(NormalMatrix_inv,K)  # Markov matrix

	del K
	del NormalMatrix
	del NormalMatrix_inv

	# print(D)
	# print(D2)
	# print(K)
	# print(N)
	# print(M)
	# print(np.sum(M,axis = 1,keepdims=True))   # Check if row sum is 1.0 for Markov matrix



	#### eigen decomposition ######

	evalu, evec = LA.eig(Markov)

	# print(w)
	# print(v)

	order = np.argsort(evalu)  # sort eigenvalue and eigenvectors as descending order
	order = order[::-1]

	evalu = evalu[order]
	evec = evec[:,order]
	# print(w)
	# print(v)

	np.savetxt('eigenvalue.txt',evalu,fmt='%.3f')
	np.savetxt('eigenvectors.txt',evec,fmt='%.3f')

	index = np.arange(1,11,1)  # select only first 10 eigenvalues to show

	# print(index)

	#### Plot eigenvalues ######

	# fig = plt.figure(figsize=(8, 6))
	# ax = fig.gca()
	# fig = plt.bar(index , evalu[0:10], 1,
	#                  alpha=0.9,
	#                  color='dodgerblue',
	#                  label='Cross-validation error ')
	# ax.set_xlabel('Eigen Index',fontsize=20)
	# ax.set_ylabel('Eigenvalues',fontsize=20)
	# # plt.ylim((0,0.2))
	# plt.savefig('eigenvalue.png',dpi=300)



	#### Plot eigenvectors ######

	# ax1 = 1
	# ax2 = 2

	# # NP = len(Pdist)
	# # print('NP=',NP)
	# length = len(evec[:,ax1])
	# print(length)

	# fig = plt.figure(figsize=(8, 6))
	# ax = fig.gca()
	# # rg = np.loadtxt('Rg.txt')
	# plt.scatter(evec[:,ax1],evec[:,ax2], c=np.arange(0,length,1),marker='.',s=60, cmap='jet')
	# # plt.scatter(v[:,ax1],v[:,ax2], c=meaning3, marker='.',s=60, cmap='jet')
	# # plt.scatter(np.arange(0,length,1),v[:,ax1], c=np.arange(0,length,1),marker='.',s=60, cmap='jet')

	# plt.colorbar()
	# ax.set_xlabel('Evec '+str(ax1),fontsize=20)
	# ax.set_ylabel('Evec '+str(ax2),fontsize=20)

	# plt.savefig('eigenvector.pdf',dpi=300)

	return evalu, evec




def nystrom(COORD_b,COORD_dmap,evl,evc,NC,epsilon,exponent,h2tflag,mirrorflag):
	Nbreak = len(COORD_b)
	Ndmap = len(COORD_dmap)

	NystromD = np.zeros((Nbreak,Ndmap))
	# compute pdist in nystrom
	print('compute pdist in nystrom, Nbreak,Ndmap=',Nbreak,Ndmap)

	for i in range(Nbreak):
		if i%100==0:
			print(i)
		for j in range(Ndmap):
			NystromD[i,j] = dist(COORD_b[i],COORD_dmap[j],NC,h2tflag,mirrorflag)

	# epsilon = 0.5

	D2 = np.power(NystromD,exponent)  # distance^2

	del NystromD # delete Pdist to release memory

	K = np.exp(-D2/epsilon)  # kernel matrix
	# print(K)

	del D2 # delete D2 to release memory

	NormalMatrix = np.sum(K,axis = 1,keepdims=True) 
	Markov = K/NormalMatrix

	nyevc = np.zeros((Nbreak,len(evl)))
	for i in range(len(evl)):
		nyevc[:,i] = np.matmul(Markov,evc[:,i])/evl[i]
	return nyevc

