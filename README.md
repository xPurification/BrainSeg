**Brain Segmentation and Alzheimer's Detection Using AI**

_Keanu Francis, Tyler Gennaci, Oliver Pennanen, Devon Purification, Tumi Shoyoye_

_Florida Atlantic University - Department of Engineering and Computer Science_

[Research Manuscript](ResearchManuscript.pdf)

**Abstract**

This project has two main focuses: the segmentation of DICOM FDG-PET scan images
and the analysis of the PET scan images to determine if the patient does, in fact, have
Alzheimer's. This process is normally time-consuming and must be done manually. Reducing the
time it takes would significantly reduce the cost of care and could hopefully extend the lives of
people suffering from this disease by enabling rapid diagnosis and treatment.

This is done with three convolutional neural networks and image processing. The first
neural network is a U-NET architecture that segments the brain from the surrounding tissue and
is also trained to remove images that contain artifacts from the scanning process. The next AI
performs binary classification on each image and gives it a value of either CN (cognitively
normal) or AD (Alzheimer's disease). The final AI segments the image into the frontal, parietal,
occipital, and temporal lobes, as well as the cerebellum

**Introduction**

PET scan imaging is one of the most potent tools used in the medical field. Positron
Emission Tomography uses injected glucose to create 3D images of the inside of the body. These
images can then be used to determine many forms of illnesses. This can be used to diagnose
cancers, brain disease and cardiovascular health. Alzheimer's is a progressive disease that slowly
destroys brain tissue leading to shrinkage of the brain and the death of nerve cells. This is the
leading cause of dementia. One method of detecting the afflicted areas is with the use of PET
scans. Professionals require years of experience to analyze the brain of different scans and
determine if that patient is in fact suffering from Alzheimer's. To better increase the accuracy of
these professionals and decrease the cost of diagnosis. A system must be created that can
perform Alzheimer's detection quickly and accurately

The purpose of our research is to develop an algorithm that simplifies the segmentation
process and predicts the likelihood of Alzheimer's disease in patients based on FDG-PET scan
images. This program aims to utilize glucose concentration within various brain structures as a
marker for Alzheimer's, leveraging the understanding that Alzheimer's patients exhibit
significant neurological degeneration and shrinkage in the brain, leading to changes in glucose
levels.

**Literature Review**

Unfortunately there is not a significant amount of progress made on segmenting PET
scan images when compared to scans such as MRI. Segmentation is usually done through image
thresholding to generate differences between the foreground and background images. However
segmentation of PET scan images run into the problem of PET scans being lower resolution with
high contrast.

The low resolution and high contrast of noise generates a significant amount of
uncertainty in the thresholding approach. These are typically in the form of Fixed Thresholding,
Adaptive Thresholding, and Iterative Thresholding.

**Segmentation Architecture**
![Segmentation Architecture](https://github.com/user-attachments/assets/79358347-a0d6-4520-8f90-b1dd6d5f2f5c)

**Discussion**

Based on our results, Alzheimer's patients demonstrate lower levels of glucose within the
brain compared to healthy patients. However, further examination of white matter glucose
concentration compared to gray matter is needed for optimal results. With an average
segmentation and Alzheimer's prediction time of 250ms CPU and 30ms GPU per slice, doctors
will be able to accurately determine the possibility of Alzheimer's within the brain far faster than
manual human analysis, leading to a reduction in cost and diagnosis time.

_Glucose Calculation_

Glucose calculations were made using the segmentation mass and the associated intensity
of each pixel of the segmented brain to find an average intensity value. When comparing the
healthy brain to such brains, we can see a drop in the average glucose in patients with
Alzheimer's disease. This, however, is mostly seen in the mid-brain where most of the white
matter is. This is also reflected in the Alzheimer's prediction.

_Alzheimer's Prediction_

From the Alzheimer's AI, we can see a vast variation in its accuracy based on whether or
not the PET scan was done accurately. We also see vast variation in the Alzheimer's prediction in
regions of the brain with consistently high glucose such as the cerebellum and the top of the
parietal lobe.

_Brain Segmentation_

Brain to non-brain segmentation proved to be highly accurate but loses some accuracy
near the beginning of the cerebellum and the transition from the cerebellum to the parietal lobe.

_Brain Region Segmentation_

Region segmentation is proven to be highly accurate. However, this accuracy is
dependent on the accuracy of the training data. A professional neurologist needs to create the
original training data. Without proper training data, the brain will be segmented accurately, but
the accuracy of said segmentation may or may not correlate to a real brain.
