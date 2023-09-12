# GRConvNet
Refactored version of the Repository used in the GrConvNet Paper. 
The original repository can be found [here](https://github.com/skumra/robotic-grasping).  

## Usage
Multiple demonstrations of the usage of the refactored code can be found in the `notebooks` folder.

## Processing chain
The refactored version is using a pipeline based approach.
The components of the pipeline are:
**Dataset**  
&darr; \<raw data\>  
**DatasetClass**  
&darr; CameraData  
**Preprocessor**  
&darr; TensorType[1, 4, 224, 224]  
**Grconvnet**  
&darr; TensorType[1, 4, 1, 224, 224]  
**Postprocessor**  
&darr; List[ImageGrasp]
**Img2WorldConverter**  
&darr; RealGrasp