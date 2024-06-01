#!/bin/bash

# this script run more faster jobs


# rm *.gro
# rm -r gros datas

# mkdir gros
# mkdir datas

#For loop in bash shell 
for i in $(seq 1001 1 5000)
do
echo "$i"
# mkdir $i
# rm -r $i
# cd $i

# create initial gro configuration
echo "0"|gmx_mpi trjconv -s ../3_eq_NVT/eq.tpr -f ../3_eq_NVT/eq.trr -o gros/eq$i.gro -pbc whole -b $i -e $i -vel no

# transfer all gro to lammps data
python /home/jw/Desktop/Alkane_Combustion/Codes/gro2data_ID.py -i inputdata.in -ID $i




done