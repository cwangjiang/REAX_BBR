#!/bin/bash

# this script run more faster jobs

# remove all existing folders

Core=1

start=1
end=1000

NC=5
coregroup=1

# for i in $(seq $start $end)
# for i in 5 8 
# do
# rm -r $i
# done



#For loop in bash shell 
for i in $(seq $start $end)
# for i in 15 165 
do
echo "$i"
rm -r $i
cp -r basic $i
cd $i

# get the i-th initial configuration
cp ../../1_Gromacs/5_create_initials/datas/data$i.ALKO2 .

# run interations using python
python ../../../Codes/runLammps.py -i $i -n $Core -cg $coregroup -nc $NC

cd ..


done