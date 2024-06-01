import numpy as np
from numpy import linalg as la
import math


def Kabsch(A,B,l):

	# compute sum of each axis
	SUMA=A.sum(axis=0)
	SUMB=B.sum(axis=0)

	# Centralize each dimension
	A=A-SUMA/l
	B=B-SUMB/l

	H=np.dot(A.transpose(),B)

	U,sig,W = la.svd(H)

	I = np.eye(3)

	d=la.det(np.dot(U,W))
	if d<0.0:
		# print(d)
		I[2,2] = -1.0


	R = np.dot(np.dot(W.transpose(),I),U.transpose())

	Diff = np.dot(A,R.transpose()) - B;
	distance = math.sqrt((Diff*Diff).sum()/l)

	return  distance









