# ANPR

## Version 4.0 (upcoming)
1. The ANPR version 4.0 is tuned to operate with flask

## Version 3.1
1. Creation of class ANPR for better management with flask

## Version 3.0
1. The ANPR version 3.0 contains code on detecting car and number plate together and match the numberplate to individual car 
2. The model is using onnx as I tried to improve the performance on intel processors but no help.
3. The current pipeline is having Wrap Transformation which is cropping the numberplates from one edge to other seamlessly.

## Version 2.0
1. The ANPR version 2.0 contains code on detecting car and number plate together and match the numberplate to individual car 
2. The ANPR is better then the last version

## Version 1.0
1. The ANPR version 1.0 contains code on detecting Number Plates with on each car
2. The ANPR is quite slow as it has to check on multiple loops if the frame car has numberplate 

# GIT Commit commands:
1. git add --all
2. git commit -m "commit message"
3. git push origin main
4. git tag vX.X
5. git push origin vX.X
