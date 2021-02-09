# Emotion Regression in the Wild
Emotion regression using estimation of Valence and Arousal values in videos available in Aff-Wild database. 
We used 2 CNN based frameworks for this problem. 
One of the models used <b> SeNet</b> pre-trained on <b>VGGFace</b> database and fine-tuned the model on a subset of the Aff-Wild train data. The other model was a <b> ResNet </b> style CNN with <b>CBAM</b> attention module for refined feature extraction. This model was trained from scratch using the subset of Aff-Wild train data.
```
The hyper-parameters used for both the models are listed below:
Batch Size    = 32
Optimizer     = Adam
Learning Rate = Default
Epochs        = 32
```
<p align="center">
The train and validation root mean square error graphs of both frameworks are shown below.
  
<img width=400 src='https://github.com/A-Janj/Emotion-Regression-in-the-Wild/blob/main/resources/CBAM_results.png'/>
<img width=400 height=350 src='https://github.com/A-Janj/Emotion-Regression-in-the-Wild/blob/main/resources/VGG_results.png'/>
<pre>                     CBAM Framework                           Transfer Learning Framework </pre>
</p>
<p align="center">
The values of Valence and Arousal were used to find a categorical emotion using the 2D Emotion (Valence-Arousal) Wheel below.
<img src='https://github.com/A-Janj/Emotion-Regression-in-the-Wild/blob/main/resources/2D%20Emotion%20Wheel.png'/>
  </p>
