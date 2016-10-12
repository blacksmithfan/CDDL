Relevant techniques utilized in this implementation:
ScSPM (http://www.ifp.illinois.edu/~jyang29/ScSPM.htm)
K-SVD (http://www.cs.technion.ac.il/~ronrubin/software.html) 
LC-KSVD (http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html)
All codes are provided for noncommercial research use
If you happen to use this code, please cite our work:
[1] F. Zhu and L. Shao, ¡°Weakly-Supervised Cross-Domain Dictionary Learning for Visual Recognition¡±, 
International Journal of Computer Vision (IJCV), vol. 109, no. 1-2, pp. 42-59, Aug. 2014.
[2] F. Zhu and L. Shao, ¡°Enhancing Action Recognition by Cross-Domain Dictionary Learning¡±, 
British Machine Vision Conference (BMVC), Bristol, UK, 2013.
If you find any bugs in this code, please contact (fan.zhu@sheffield.ac.uk)


Installation:

1. Pre-compiled mex functions are provided for 64-bit windows systems. If you are using other types of OSs, 
please compile C files in the private folder of OMPbox and ksvdbox first.
2. If you have not done so before, configure Matlab's MEX compiler by entering
    >> mex -setup
	and follow the instructions.
	
Usage:
1. For a quick start, you should copy your data folders (both target domain and source domain) into the 
``image'' folder, where the ``image'' locates under the root directory. The standard format of your data
folder should look as follows:
<folder_name> <-- <image_categories> <-- <images_belonging_to_each_category>

2. Set para.dataSet in line 22 of main.m file as your target domain folder_name, and set para.dataSet in 
line 59 of main.m file as your source folder_name.

3. Annotations of parameter settings are provided in the main.m file.

4. If you are working on video data, or would like to use other image features, you'll have to massage this
code to fit your features.