# Point Cloud Completion of Articulated Objects and AAPCC Dataset
This repository contains the partial implementation of a deep-learning-based point cloud completion algorithm design for articulated objects (especially for articulated animals) and the testing sets of an newly developed Articulated Animal Point Cloud Completion dataset. The quantitative evaluations of the algorithm can be done with these codes. More codes will be uploaded.

## 7 Steps to Run the Codes
1, Install "torch==1.8.1", "torch_geometric==2.0.4", "tqdm==4.64.0", "numpy==1.22.3", "trimesh=3.10.8".

2, Download the testing sets of the AAPCC dataset at "https://1drv.ms/u/s!AryidJO66al4jF2YE1-MhFxnLkwG?e=WHKRXk". Extract them into the "data" folder.

3, Download the hyper-parameter data required by the algorithm at "https://1drv.ms/u/s!AryidJO66al4jF7erRUzJNk4nH9O?e=txysrf". Extract them into the "evaluation/ours/spiralae" folder. This data is generated based on existing codes.

4, Download the fixed mesh topology of the SMAL model at "https://1drv.ms/u/s!AryidJO66al4jFw73ON4zlRCb9Qc". Move it into the "evaluation" folder.

5, Download the trained parameter data at "https://1drv.ms/u/s!AryidJO66al4jF_s3iMD5_8R67xj?e=IFnRKB". Extract them into the "main/trainedOursAAPCC" folder.

6, Edit line 135-137 and 186 of the source file "evaluation/ours/quantitativeEval.py" following the comments.

7, Run "quantitativeEval.py" like "python quantitativeEval.py ../../main/trainedOursAAPCC/train_200_1_split3_41.pth".
