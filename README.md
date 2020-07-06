# ImageProcessing_Panorama

In this exercise i've implemented automatic ”Stereo Mosaicking".

First a video is splitted info frames.

The homography between frames is calculated with Harris corner detector.
MOPS descriptors are extracted using gaussian pyramids among others. 
The descriptors are matched using RANSAC algorithm among others...

After all the panorama is getting stitched all toghether.

Notes:

The class PanoramicVideoGenerator and the file my_panorama.py was given and was not written by me.
