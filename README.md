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


&darr; (List of grasps, camera_position, camera_intrinsics, camera_rotation)  
**Img2WorldConverter**  
&darr; RealGrasp
**GraspHeightAdjuster**
&darr; RealGrasp

## Design Decisions
- To simplify the use of different datasets the all dataset should return an instance of (subclass of) CameraData
    - the CameraData base class holds a number of properties which might be used by the preprocessing pipeline.
    - if the original dataset does not provide the necessary information for some of the CameraData base class, reasonable defualts must be provided
    - any additional information which might be contained by the dataset can be kept by using a subclass of CameraData which extends their properties.
- the CameraData classes, whoses instances are obtainedby the dataset classes, should have their images saved as Tensors, not as PIL images. 
    - This allows batched preprocessing with the T.Compose pipeline for "linear" pipelines. (see https://discuss.pytorch.org/t/can-transforms-compose-handle-a-batch-of-images/4850/5) 
    - the image tensors should be in (c,h,w) format to be consistent with the torch API
    - also the images should be in range ... and dataformat torch.FloatTensor
- the upside of using wrapping pipelines is that we can easily use the same pipeline in different cases and be sure that they are the same. The downside is that we might need to access intermediate results of the pipeline which are not returned for analysis and sometimes the parametrization can become to verbose.
    - to allow usage of granular and wrapped pipelines the processing folders are seperated in "custom_transforms" and "pipelines"
    - to access intermediate steps of a wrapping pipeline intermediate results can be saved into the properties of the wrapping pipeline at each \__call\__ 
    - "custom_transforms.py" are the elementary building blocks for the pipelines in "pipelines.py". Intermediate sub-pipelines might be implemented in "pipelines.py" but might make the code harder to understand and are therfore discouraged.
    - if a custom transform is a oneliner and has no configurable parameters its not worth creating it, simply use the according function
    - a complete preprocessing pipeline makes sense as we need the same preprocessing for different applications, however a complete postprocessing pipeline is not that beneficial as we do not always have camera intrinsics etc
    - an utility end2end processor is implemented whcich combines all possible steps
    