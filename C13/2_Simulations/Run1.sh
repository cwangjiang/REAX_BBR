#!/bin/bash

# this script run more faster jobs


Core=2

# remove all existing folders
start=4001
end=4250
NC=13
coregroup=1

# for i in $(seq $start $end)
# do
# rm -r $i
# done



#For loop in bash shell 
for i in $(seq $start $end)
# for i in 2010 2232 2378 2476 2489 2533
do
echo "$i"
rm -r $i
cp -r basic $i
cd $i

# get the i-th initial configuration
cp ../../1_Gromacs/5_create_initials_1000K/datas/data$i.ALKO2 .

# run interations using python
python ../../../Codes/runLammps.py -i $i -n $Core -cg $coregroup -nc $NC

cd ..


done