# GRConvNet
Refactored version of the Repository used in the GrConvNet Paper.   

## Processing chain
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