# NASA_Hackathon2020_Team10
Team10 - `Library Master`   
Challenge: `Scanning for Lifeforms`   
Keyword: `IoT`, `Few-shot Learning`, `Model Compression`, `NeuroPilot`, `Biodiversity exploration`

## Introduction
Many previous research works have shown the effeciency of Few-shot learning when lacking of labeled data.
In our challenge, `biodiversity exploration for rare species`, we are facing the dilemma which is similar to Few-shot learning situation.
Therefore Few-shot Learning strategy may classify more precise among common species and rare one than naïve classification methods.
- To evaluate this speculate, we have done several experiments to show the advantages of Few-shot learning.
- In order to improve the performance when implementing on edge devices, we also show the model compression work we have done.
![](https://i.imgur.com/Lex38aU.png)

## Reproduce details
- [naïve classification](https://github.com/summelon/NASA_Hackathon2020_Team10/blob/main/naive_classification/README.md)
- [Few-shot learning](https://github.com/summelon/NASA_Hackathon2020_Team10/blob/main/few_shot_learning/README.md)
- [Edge implement](https://github.com/summelon/NASA_Hackathon2020_Team10/blob/main/edge_implement/README.md)

## Main Results
- The figure(day view) on the left shows the superior of model trained on Few-shot learning, the right one(night) shows the further impact compare to naïve classification model
- The table shows that our model compression work significantly improve model inference time and power consumption by model quantization
![](https://i.imgur.com/5gYe8jh.png)

## Reference
- [Deep Learning for Large Scale Bio-diversity Monitoring](https://conservationmetrics.com/wp-content/uploads/Klein_2015_bloomberg_data4good-2015.pdf)
- [A deep active learning system for species identification and counting in camera trap images](https://www.researchgate.net/publication/336735839_A_deep_active_learning_system_for_species_identification_and_counting_in_camera_trap_images)
- [Scene‐specific convolutional neural networks for video‐based biodiversity detection](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13011)
- [Regularizing Meta-Learning via Gradient Dropout](https://arxiv.org/abs/2004.05859)
