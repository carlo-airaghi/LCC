# Thesis

## Abstract

Deep networks have become state of the art for solving many computer vision tasks. Land cover classification problems are no exception, with the convolutional neural networks excelling in performing the pixel-by-pixel classification. Nevertheless, their usage brings on the table new challenges with respect to traditional methods. Indeed, deep network models can deal only with inputs of limited size (e.g. 512×512) to control an exponential growth of the models' training time and memory occupied by the models' variables. Dealing with remote sensing (RS) images this limitation is very binding. RS images are indeed of considerably larger dimensions, and must be split into small patches (hundreds) to be fed into the models. This splitting operation opens to a few side effects, such as the limitation of the receptive field of the model, which affects the overall accuracy of the predictions. This research investigates whether those issues could be solved with parallel combinations of simple CNN-based models. The results imply that an effective method to address those problems is an architecture aligning in parallel models trained with images taken with different proportions of ground resolution and land coverage.

## Keywords

Land cover classification, pixel-level, feature fusion, remote sensing, forest cover

## Introduction

The presently available technologies for earth observation produce a large quantity of high-resolution remote sensing images. Images that government programs such as ESA’s Copernicus, NASA’s Landsat and CNSA's Gaofen are making freely available for commercial and non-commercial purposes with the intention of fueling innovation and entrepreneurship. In the last few years, this data allowed researchers to address a large number of problems in the fields of agriculture [1][2][3], geosciences [4], and climate change [5][6]. Many of those problems are faced starting from the fundamental task of classifying the land cover, in order to have insights about what are the characteristics of the environment and how it is changing over time.

The land cover classification problem is a subset of the more general field of semantic segmentation; a field that in the last few years flourished thanks to the improvement of new machine learning tools such as deep neural networks. Even though it has already been proved how successfully the land cover classification issue can be solved using Deep Networks(DN) [7], to address it with a higher accuracy it's needed to consider some peculiarities of the data the segmentation is applied to.

Usually, remote sensing images with respect to typical natural images used in computer vision applications have a considerably larger size. Specifically, to make the training of common deep networks computationally affordable, those are fed with images with dimensions smaller than 512×512 pixels; RS images on the other hand can have sizes even wider than 10000 pixels per dimension.

Being computationally unaffordable to train and test classifiers on these images at a full-size scale, the state-of-the-art methods suggest splitting the RS images and extract small patches suitable for training the DN model [8].

This methodology (which will be further explained in section 2.1 and a visual representation can be found in Figure 1) is effective for training the DN. But while performing predictions in a complete RS image the process of cropping, predicting and merging the predicted patches, introduces two issues.

The first problem is the multiplication of image borders during the splitting process. DN models for semantic segmentation, such as UNet++ [9] and DeeplabV3 [10] that will be used in this research, are made up primarily of multiple convolutional layers. In those layers, a kernel of given size “slides” over the 2D input data (a patch), performing an element-wise multiplication. The multiplication results are then summed up into a single output pixel. The kernel will perform the same operation for every location it slides over, transforming the input 2D matrix of features into a different 2D matrix of features [11].

As shown in Figure 2, if the kernel has dimensions wider than 1×1, it requires the application of a zero-padding method (or similar approaches) before multiplying the kernel and the feature map on the pixels laying on the border of the image.

Irrespective of how ingenious the use of methodologies such as zero-padding, the predictions that a model performs along the edges of an image are less accurate than the ones performed on the center of the image. As a matter of fact, while those last are based purely on the overlapping of the kernel on an authentic matrix being a portion of the input, the ones on the edges are the result the products between the kernel and matrixes that are partially coming from the input and partially made up. Since the RS images have to be split for performing the semantic segmentation, there is an increase in the number of borders, and therefore a propagation of the number of the predictions carried out on partially made up matrixes. In particular, it can be identified a place of points (pixels) in an image, being a grid with the distance between the lines equal to the dimension of the patches cut out from the image; where the described accuracy decrease occurs.

To deal with this issue, in section 3.2, we propose an architecture aiming to make the predictions less dependent on the way the original RS images are cropped, and in section 4.2 we show the performance of that architecture.

The second problem introduced by the process of cropping, predicting and merging the prediction results is the incompleteness of the information included in a single patch. And having patches including incomplete (cropped) objects makes the segmentation tougher. In typical segmentation applications (example in Figure 3), most of the features that have to be classified are completely included within the boundaries of the image.

In RS images, the classification target: morphological features of the land cover, can have highly variable dimensions. A woodland, for example, can be a wood covering a small area as a forest of hundreds of thousands of square kilometers. With the dimension of a patch being 512×512 that with a 10-meter ground resolution leads to an area of 25 thousand square kilometers. That area could be limited in a few conditions and not be sufficient to fit an entire morphological feature in a single patch. Thus, a DN based classification over RS images can't rely on the borders of a feature, but must be based only on the colors of the pixels and patterns included on the single patches. A trivial solution would be to downscale the RS image before cropping to obtain patches less detailed but covering wider areas, that would however lead to prediction with lower resolution and would imply an intolerable loss of information.

To face this issue, in section 3.3 we propose an architecture that combines the level of detail of the predictions performed on RS images at the original dimension with predictions performed on patches coming from downscaled RS images yielding wider contextual features. Architecture whose performance is tested in section 4.3.

This research aims to estimate the feasibility of those two architectures meant to solve the previously exposed problems and evaluate their performances.

## Related Works

### Land cover classification

With the improvement of spatial resolution of remote sensing (RS) images, RS image classification gradually formed three parallel classification branches at different levels: pixel-level, object-level, and scene-level classification [12] whose examples are shown in Figure 4.

The pixel-level classification task consists in the labeling of each pixel in the remote sensing images with a semantic class. Essentially the pixel-level classification in the remote sensing field corresponds to the segmentation tasks in the computer vision word. This branch was the central topic of the early LCC literature. Researchers mainly focused on classifying remote sensing images at pixel-level [13] since the spatial resolution of remote sensing images was very low and the size of a pixel was similar to the sizes of the objects of interest. With the increase of the level of detail of remote sensing images captured by satellites, the scientific community interest moved also on the possibility of identifying smaller and more specific objects in the ground (object-level classification) and merging the information linked with those objects to attribute semantic classes to remote sensing images.

Nevertheless, the pixel-level classification remained a subject of great interest for the researchers since it allows to monitor the landscape change detection and permit land use/cover classification. Thus, it will be the kind of LCC on which this research focuses.

### Remote sensing segmentation techniques

RF and SVM are the most-common classic classifiers used in literature for the land cover classification of remote sensing data. Like the other non-parametric supervised classifiers, these algorithms do not make any assumption regarding the distribution of data and they have shown promising results in classifying remote sensing data overtaking the field’s earlier classifiers adopted such as Linear Regression (LG), Maximum Likelihood (MLC), K Nearest Neighbor (KNN) and Classification and Regression Tree (CART) [14]. However, in the last few years, deep learning models such as CNNs have been proved to be very successful in classifying complex contextual images, and have been widely used to classify remote sensing data, outperforming the classic methods [7]. Between those, U-Net++ and DeeplabV3+ stand out for performance. Moreover, those two are simple to implement even with modest resources because their weights pre-trained on ImageNet [15] are readily available; this reduces significantly the successive training time on RS images. Thus, they have been used as fundamental blocks in our new architecture presented in section 3.3.

### U-Net++

Unet [16] in Figure 5 is a symmetric architecture consisting of two major parts. The left part is called the contracting path, which is constituted by the usual convolutional process: a sequence of down-sampling blocks, each consisting of a sequence of 2 convolutional layers with a 3×3 kernel size and a padding of 1 pixel. The first element of each block results from the application of the max-pooling layer (with a 2×2 kernel and stride of 2) on the result of the previous block. The right part is the expansive path, which is constituted by transposed bidimensional convolutional layers performing an up-sampling on their respective input. The interesting feature of UNet is the input of the up-sampling blocks, which is not made only from the result of the previous block, but also from the down-sampling blocks on the left side having the corresponding output dimension. Those “skip connections” allow the contextual information typical of less-processed images to flow into the convolutional layers without losing the localization accuracy caused by the down-sampling of the max-pooling steps. UNet++, proposed by Zhou et al. in 2018 [17], is an evolution of UNet succeeding to improve the accuracy of the previous architecture by adding more intermediate (nested) convolutional blocks and densifying the skip connections between blocks. Following the thesis of Zhou, having a denser structure implies an increased semantic similarity between the results of each couple of concatenated up-sampling blocks, resulting in an easier optimization problem to solve in the training and therefore a higher overall accuracy.

### DeeplabV3+

The Deeplab architecture [10], as well as its successive evolutions, has been proposed by Liang-Chieh Chen and the Google team [18]. To date, four versions have been launched. The last release of the network, DeeplabV3+, likewise UNet has an encoder-decoder structure with the encoder being based on Xception [19] with a number of variations proposed by Liang-Chien Chen. In terms of performance, two of the most impactful additions to DeeplabV3+ were the Atrous Convolution (AC) and Spatial Pyramid Pooling (SPP). AC allows including a larger receptive field in the prediction while SPP makes the model capable of carrying out equally accurate predictions on features of different scale.

In more detail, the Atrous convolution (in Figure 6) is a convolution where the kernel is implemented with an internal spacing (stride) given by a parameter "rate". With AC the kernel samples the input values every rate pixels in the height and width dimensions. Thus it receives as input a larger receptive field without letting the resolution decrease. Moreover, AC is computationally lighter than the standard convolution and makes the models which are making use of it way faster.

The Spatial Pyramid Pooling (SPP), which was first proposed by He et al [20], is a building block typical of the Deeplab architectures, which substitutes the single pooling layer commonly used at the end of a NN between the last convolutional layer and the fully connected layer. SPP resamples the given feature layer at multiple rates prior to convolution. Thus, the feature map is divided into a fixed number of spatial bins having sizes proportional to the image size. Each bin gives a different scaled image (see Figure 7). The bins are eventually flattened and given as input the fully connected layer. This, in contrast with the AC, adds learning complexity and slows down the training process, but makes the model capable of coping with features of different scales.

Clearly, both models were designed with the aim of improving the receptive field and thus, to make the resulting predictions more dependent on the input images as a whole. This research intends to follow this same principle, applied to the domain of remote sensing, and proposes a solution to extend it with the parallelization of more models performing predictions at different levels of ground resolution.

## Data

The North Holland province, with the website "Copernicus Open Access Hub" [21], has access to RS images collected from the ESA's Copernicus Sentinel-2 mission. From those images, coming in different formats and bandwidths, the more appropriate for the LCC problem are the ones having a higher resolution (10 meters per pixel) and capturing all the visible features of the land cover.

The hub, considering the short revisit time of the Sentinel's satellites [22] and that the mission started making the data public in December 2016, provides a huge amount of those images. Nevertheless, those data are **unlabelled**, and to train the model a ground truth is needed. Therefore it has been necessary to look for an external labelled dataset of RS images similar to the Sentinel's. Most of the available annotated datasets were inadequate to fulfil the task though. E.g. the Sentinel-2 level 1c classification, the most straightforward choice since it is performed over the Sentinel-2 data itself [23], has only one generic label to describe the vegetation. Thus, a model trained on those data cannot distinguish between different types of vegetation and is therefore inadequate for the province purpose of monitoring the mutations of the forest.

Moreover, the research aims to address the issue using Deep Networks [7], and those require large labelled datasets to be successfully trained and reach high accuracies.

A dataset that is sufficiently large and shares a high degree of similarity to the Sentinel2 is the GID dataset [24].

The GID dataset is a **labelled** dataset collected from the Chinese satellite Gaofen-2. Despite having different construction, and operating in a different way to the satellites of the Sentinel-2 mission, GF-2 gathers data that are sufficiently similar to those of the Copernicus mission.

While creating the GID dataset Tong et al. [24] produced two different types of pixel-level annotation: a large-scale classification set and a fine land-cover classification set (example in Figure 8). For the large-scale, they annotated 150 GF-2, and for the fine LCC set they labelled 10 GF-2. GID is widely distributed over geographic areas covering more than 70,000 km². Benefiting from the various acquisition locations and times, GID presents rich diversity in spectral response and morphological structure. For the large-scale dataset five representative land cover categories are selected to be annotated: built-up, farmland, forest, meadow, and waters. Areas that do not belong to the above five categories are labelled as unknown (or background). On the other hand, albeit having a smaller amount of annotated data the fine LCC set comprises 15 mask categories: industrial land, urban residential, rural residential, traffic land, paddy field, irrigated land, dry cropland, garden land, arbour forest, shrubland, natural meadow, artificial meadow, river, lake and pond.

In the training process of architectures we proposed, loss of information is intrinsic because of the required down-scaling of the satellite images (details will be explained in section 3). Therefore, in order to have a sufficient amount of data to guarantee a proper training of the DN, it has been chosen the large-scale GID dataset.

## Proposed Architectures

### Standard architecture

![Standard Architecture](Images/Standard_Architecture.png)

As introduced in section 1, to perform the semantic segmentation on remote sensing images through the use of deep learning models, it is needed to preprocess the input of the model and split it into small patches. In Figure 1 is illustrated the common way the input is processed before the prediction and merged afterwards. A similar architecture has been proposed by Liu et al [8], whose also proposed a partial overlapping of the patches. Here we describe an even simpler version, without overlapping, which will be used as benchmark for the performances of the other architectures we proposed and analysed.

The satellite images of dimensions 7168×6656×3 (width, height, channels) are split into patches of dimension 512×512×3.

The patches are given as input to a model chosen as a baseline, either UNET++ or Deeplabv3 as explained in the related works section. Which is pre-trained on IMAGENET and further trained on patches of the same dimension coming from the GID dataset. From now on this trained model will be called M1.

M1 acts by mapping each pixel of the input patch with a probability vector stating the probability with which it belongs to one of the 6 classes the GID dataset is labelled with. Therefore, the output is a matrix of dimension 512×512×6, with the dimensions being respectively width, height and class.

The model's outputs are collected and merged back into a sole prediction matrix with dimensions (7168×6656×6) coherent with the original dimension of the RS image.

Eventually, applying the Argmax function to the prediction matrix, along the class axis, is obtained a bidimensional matrix of dimension (7168×6656) having in each pixel a number corresponding to the class it belongs to.

This standard approach, though, introduces two challenges:

- The predictions performed on the borders of the patches are less accurate than the ones done in the center of the patches.
- It's hard to reach high accuracies because of the loss of context features while focusing the predictions on small areas.

### Staggering cropping

![Staggering Cropping Architecture](Images/Staggering_Cropping.jpeg)

As explained in the Introduction, the models rich in convolutional layers such as DN models have a slight performance decrease while making predictions along the borders of the matrixes given to them as input. And while splitting an image in a large number of patches there is a propagation of this issue. Here, we propose an architecture aiming to prevent possible inaccuracies decoupling the splitting process from prediction outcome.

The architecture, illustrated in Figure 9, consists of two prediction pipelines paired in parallel.

The first pipeline of the architecture is equal to the standard architecture except for the absence of the Argmax function. Then, the input is an RS image of dimension (7168×6656×3) and the output is a matrix of probability vectors of dimension (7168×6656×6).

The second pipeline, on the other hand, has a few differences from the standard one.

The input is taken cropping the original input RS image such that is left an offset of 256 pixels from all the borders of the image. Therefore is obtained an input image of dimension (6656×6144×3).

The image is then split in patches of dimension (512×512×3). Thanks to the previous step, the patches obtained don't have any border in common with the patches got from the first pipeline.

The patches are then processed from M1.

The model's outputs are collected and merged back into a sole prediction matrix with dimensions (6656×6144×6).

Both those matrixes are upscaled, in the respective pipelines, to matrixes P2 and P3 of dimensions equal to the first stage output.

Eventually, the mean is taken of the probability vectors matrices of the outputs of the two parallel pipelines. The mean is then given as input to an Argmax function, applied along the class axis, leading to a bidimensional matrix of dimension (7168×6656) as overall output.

### Multi-Stage fusion architecture

![Multi-Stage Fusion Architecture](Images/Multi_stage_Architecture.png)

To deal with the issue of the loss of context features resulting from the splitting in patches of the RS images is proposed a second architecture (in Figure 10). Its purpose is to increase the overall prediction accuracy avoiding the trade-off between predictions on images with higher resolution and images less defined but covering wider areas on the ground.

The architecture can be divided into three pipelines. The first one is exactly similar to the standard architecture with only the difference being that the output P1 is not processed from the Argmax function.

The second and the third pipeline, on the other hand, are similar to the standard architecture, but they are processing different data.

Respectively in the second and in the third pipeline the original images of size (7168×6656×3) are downsized with scale to size 2 to (3584×3328×3) and with scale 3 to (2389×2218×3).

The downsized images are split in patches of dimension (512×512×3). The patches of the second stage will have a ground resolution of 20 meters and a covered area of 1024×1024 km² and the patches of the third stage will have a ground resolution of 30 meters and a covered area of 1536×1536 km².

The patches are given as input to the models M2 and M3. These models are trained with the training set portion of the GID dataset, where the images have also been downscaled accordingly to the expected dimension of the input of the models and split into patches of dimension (512,512,3). The downscaling is performed with the nearest neighbour as interpolation criteria.

The outputs of the models are a collection of patches that, as in the first stage, are merged. Therefore, stage 2 leads to a matrix of prediction of dimension (3584×3328×6) and stage 3 to a corresponding matrix of dimension (2389×2218×6).

Both those matrixes are upscaled, in the respective pipelines, to matrixes P2 and P3 of dimensions equal to the first stage output.

Eventually, the outputs are leading to three distinct matrices of prediction vectors resulting from three models trained with different levels of ground resolution and covered area.

That information is joined by summing the matrices weighted with three different weights w1, w2 and w3, one for each stage.

\[P = w1 \cdot P1 + w2 \cdot P2 + w3 \cdot P3\]

An Argmax function (applied on the class axes) is then fed with the matrix M and returns an overall output being a bidimensional matrix of dimension (7168×6656) having for each pixel the number relating to the class to which it belongs.

## Results and Discussion

### Dataset splitting

Before proceeding with the training of the M1, M2 and M3 the dataset has been split into three sets: training, validation and test having a respective fraction with the overall GID dataset of 80%, 10% and 10%.

- The training set has been used to train the models.
- The validation set to measure first the performance of each algorithm and then to tune the weights [w1, w2, w3] of the fusion of different stages, and also to assess the utility of the staggering architecture.
- The test set was eventually needed to return an unbiased metric of the overall performance of the architecture.

### Data Processing

The GID dataset is distributed from its creators as a directory containing two folders: one with the RS images and one with the corresponding labelled images. Those last are images with three channels (RGB) because each label is represented as a colour:

- black: not classified
- red: built-up land
- green: farmland
- light blue: forest
- yellow: meadow
- blue: water

To make the dataset lighter the labelled images are label encoded, meaning that the 3 channels are mapped into a single channel with values integers and constrained between 0 and 5: one for each possible class.

Then the GID dataset is further processed such that a new directory is created for each level of ground resolution associated with the stages of the Multi-Stage fusion architecture.

The first directory D1, thus, will contain two folders, one having the 512×512×3 patches obtained splitting the images having the original ground resolution, and the other one including the correspondent labelled patches of dimension 512×512 obtained cropping the one label encoded ground truth data. The second and the third directories (D2 and D3) are made of patches of the same size taken after downscaling the RS images and the labelled images with a respective dimension of 1/2 and 1/3 compared with the original GID data.

### Implementation details

Each of the Multi-Stage fusion architecture’s stages requires a semantic segmentation model trained on a dataset of patches of the stage-specific ground resolution. Therefore, the model performing predictions on the first stage, M1, is trained on patches with a ground resolution of 10m, the training set of D1. Similarly, the models M2 and M3 are trained respectively on patches of ground resolution 20m and 30m, being the training sets of D2 and D3.

The architecture of the models M1, M2 and M3 is Unet++ or DeeplabV3+ (both are tested). Regardless of the type of base architecture, the models are pre-trained on ImageNet [15] to speed up the training process. Moreover, to make the training processes of the two models comparable, both of the architectures have been trained with a batch size of 8 and using as optimizer Adam, which is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. Adam requires the following parameters as inputs:

- α: Step size parameter (set to 0.001)
- β1: Used for decaying the running average of the gradient (set to 0.9)
- β2: Used for decaying the running average of the square of gradient (set to 0.999)
- ϵ: Meant to prevent Division from zero error. ( set to 10⁻⁸)

Driven to the same purpose of confronting the results of the training of each model (M1, M2 and M3 for based on either DeeplabV3+ or Unet++), they have been trained for a fixed time of 24 hours on an NVIDIA T4 GPU of Google Colab pro.

### Metrics

The objective while training the models M1, M2 and M3 was to minimise the Dice Loss, or rather to maximise the Dice coefficient. Which is metric commonly used in semantic segmentation applications to judge the bounty of the pixel-by-pixel predictions. The Dice coefficient (in Figure 11) measures the overlap between the masks labelled from the model and the ground truth's ones.

Each class \(i\) of the GID dataset (section 2), except for the background class, is then evaluated with the \(Dice\:loss_i\) and the number of pixels \(p_i\) of the ground truth belonging to that class. Then, as overall accuracy metric is calculated the \(multiclass\:Dice\:loss\): the weighted average of the Dice losses of every single class.

\[Multiclass\:Dice\:loss\:=\:\frac{\sum{Dice\:loss_i}*p_i}{\sum{p_i}}\]

### Standard architecture results

In order to give a benchmark to the result, we tested the models M1, M2 and M3 with an architecture commonly used for using deep learning models in the remote sensing field, such as the one described in section 3.1. Thus, we tested the models on validation. In the following table are shown the resulting Multiclass Dice coefficients for each possible combination of stages of the Multi-Stage fusion architecture (thus ground resolution of the data) and type of model the architecture is based on.

|  | stage1 | stage2 | stage3 |
| --- | --- | --- | --- |
| UNet++ | 79.4% | 77.4% | 71.1% |
| DeeplabV3+ | 82.0% | 78.1% | 77.5% |

**Table 1: Standard architecture multi-class Dice coefficients for each combination of models - levels of ground resolution**

With similar hardware and fixed the training time, DeeplabV3+ reaches higher accuracies than Unet++ for each level of ground resolution of the training set.

Those results make clear how the scarcity of data is a fundamental problem while dealing with the land cover classification task. In general, it is challenging for the model to learn a variety of morphologic features of the land cover from a dataset of 150 images even though they are covering a wide area on the ground. This is particularly clear for the models M2 and M3 which are suffering from underfitting.

This is because the training set is composed of 120 images, which for the second and the third stage of the multi-layer architecture are down-scaled to a lower resolution. The down-scaling leads to a smaller number of patches per image, indeed, while M1 is trained on 120*182 patches, M2 is trained on 120*46 patches and M3 is trained on 120*20 patches. Thus, another piece of evidence carrying to this observation of underfitting is that, independently from the chosen architecture (UNet++ or DeeplabV3+), the accuracy of M1 [79.4%, 82.0%] is way higher than M2's [76.4%, 78.1%], which is way higher than M3's [71.1%, 77.5%]. Such a variation would not have been as large if it was not for the dependency of the performance on the dimension of the training set.

This statement could appear in contrast with the general, but still moderate, loss of performance observed while experimenting with the architecture on the test set. Nevertheless, this seeming contradiction finds an explanation again in the scarcity of the data available. Indeed, as we split the GID dataset, made of 150 RS images, with an 80%, 10%, 10% proportion (respectively training, validation and test set), the test set is limited to only 15 images. The patches inside each image are often strongly correlated, therefore if there are one or a few images inside of the test set coming from previously unseen landforms which are badly classified from the models they will have a heavy influence on the average Dice coefficient. Those observations suggest that for further analysis of this architecture either to start from a bigger dataset or at least to exclude from the training and testing process those RS images which differ too much from the ones the architecture will be supposed to classify while used in. a practical application.

Another consideration coming from the observation of the results for the single models in the context of the standard architecture, is the effectiveness of the peculiar building blocks of DeeplabV3+ in the LCC application. Indeed, thanks to SPP and the atrous convolution (explained in section 2.4), DeeplabV3+ is particularly suited for images containing features of different scale and outperforms Unet++ for each ground resolution of the training set.

### Staggering cropping results

After analysing the Standard architecture accuracy which provides a performance benchmark for the other proposed architectures, we studied the staggering cropping. The model M1 is used as a fundamental block for performing the predictions in the two parallel pipelines of the staggering architecture. The latter is then tested on the validation set and the results are compared with the ones of the so-called standard architecture.

|  | Standard Architecture | Staggering Cropping |
| --- | --- | --- |
| UNet++ | 79.4% | 79.0% |
| DeeplabV3+ | 82.0% | 82.0% |

**Table 2: Comparison of the multi-class Dice coefficients between the Standard architecture and Staggering cropping architecture with the model M1 being either UNet++ or DeepLabV3+**

The results in table 2 show that the addition of a second prediction pipeline in parallel to the standard one with the aim of decoupling the labelling performed by the model from the way the original RS images are cropped, did not yield the desired improvements, leaving the Dice loss almost unchanged. The results prove the ineffectiveness of the staggering architecture.

This outcome can be explained by observing that the proposed solution mainly influenced the predictions performed along the patches' borders. Even considering the border-pixels in an image processed in the standard architecture from the M1 model, the one requiring the input image to be split in the higher number of patches, their percentage is negligible over the total number of pixels of the image. Going into the specific numbers, each patch has 2044 border pixels, which, multiplied by the number of patches per full resolution RS image from the GID dataset, 182, means a total of 372008 pixels that
