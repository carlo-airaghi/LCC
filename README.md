
\documentclass[sigconf]{acmart}
%%\documentclass[manuscript]{acmart}

\bibliographystyle{unsrt}

\AtBeginDocument{%
  \providecommand\BibTeX{{%
    \normalfont B\kern-0.5em{\scshape i\kern-0.25em b}\kern-0.8em\TeX}}}
    
%%remove reference format
\settopmatter{printacmref=false}
\usepackage{eurosym}

%%table pakage
\usepackage{array}

\setcopyright{none}
\acmConference[]{Thesis}{Amsterdam, Netherlands}{}
\acmBooktitle{Multi-Stage Multiscale Training Architecture for Segmantic Segmentation of Remote Sensing Images, Amsterdam, Netherlands}
\acmPrice{}
\acmISBN{}
\acmDOI{}

\setlength{\textfloatsep}{0pt plus 1.0pt minus 1.0pt} 

%%\acmSubmissionID{123-A56-BU3}
%%\citestyle{acmauthoryear}

\begin{document}
%
%\title{Multi-Stage Multiscale Training Architecture for segmantic segmentation of remote sensing images}

%\author{\textbf{Author:} Carlo Airaghi}
%\email{carlo.airaghi96@gmail.com}
%\affiliation{%
% \institution{Universiteit van Amsterdam}}


%author{\textbf{ Supervisor:} Qi Bi}
%\email{q.bi@uva.nl}
%\affiliation{%
% \institution{Universiteit van Amsterdam}}
 
% \author{\textbf{ Supervisor:} Jeroen Silvis}
%\email{jeroen.silvis@noord-holland.nl}
%\affiliation{%
%\institution{DataLab, North-Holland Provincie}}
%%

\begin{teaserfigure}
  \includegraphics[width=\textwidth]{Images/Amsterdam_ Teaser.jpg}
  \caption{Figure 1: Satellite image of Amsterdam taken from Copernicus Sentinel-2B}
  \label{fig:teaser}
\end{teaserfigure}

\begin{abstract}
Deep networks have become state of the art for solving many computer vision tasks. Land cover classification problems are no exception, with the convolutional neural networks excelling in performing the pixel-by-pixel classification.
Nevertheless, their usage brings on the table new chellenges with respect to traditional methods. Indeed, deep network models can deal only with inputs of limited size (e.g. 512$\times$512) to control an exponential growth of the models' training time and memory occupied by the models' variables. Dealing with remote sensing (RS) images this limitation is very binding. RS images are indeed of considerably larger dimensions, and must be split into small patches (hundreds) to be fed into the models. This splitting operation opens to a few side effects, such as the limitation of the receptive field of the model, which affects the overall accuracy of the predictions.
This research investigates whether those issues could be solved with parallel combinations of simple CNN-based models. The results imply that an effective method to address those problems is an architecture aligning in parallel models trained with images taken with different proportions of ground resolution and land coverage.

\end{abstract}

\keywords{Land cover classification, pixel-level, feature fusion, remote sensing, forest cover}

%\begin{teaserfigure}
%  \includegraphics[width=\textwidth]{Image%s/Amsterdam_ Teaser.jpg}
%  \caption{Figure 1: Satellite image of %Amsterdam taken from Copernicus %Sentinel-2B}
%  \label{fig:teaser}
%\end{teaserfigure}

%%set page numbers, not in default template
\settopmatter{printfolios=true}

\maketitle


\section{Introduction}
\label{sec:intro}
The presently available technologies for earth observation produce a large quantity of high-resolution remote sensing images. Images that government programs such as ESA’s Copernicus, NASA’s Landsat and CNSA's Gaofen are making freely available for commercial and non-commercial purposes with the intention of fuelling innovation and entrepreneurship. In the last few years, this data allowed researchers to address a large number of problems in the fields of agriculture \cite{flood}\cite{soil-moisture-variability}\cite{agriculture-sentinel}, geosciences\cite{wave-height} and climate change\cite{greenhouse}\cite{climate-change}. Many of those problems are faced starting from the fundamental task of classifying the land cover, in order to have insights about what are the characteristics of the environment and how it is changing over time. 

The land cover classification problem is a subset of the more general field of semantic segmentation; a field that in the last few years flourished thanks to the improvement of new machine learning tools such as deep neural networks. Even though it has already been proved how successfully the land cover classification issue can be solved using Deep Networks(DN) \cite{DNN-LULC}, to address it with a higher accuracy it's needed to consider some peculiarities of the data the segmentation is applied to.

Usually, remote sensing images with respect to typical natural images used in computer vision applications have a considerably larger size. Specifically, to make the training of common deep networks computationally affordable, those are fed with images with dimensions smaller than 512$\times$512 pixels; RS images on the other hand can have sizes even wider than 10000 pixels per dimension. 

Being computationally unaffordable to train and test classifiers on these images at a full-size scale, the state-of-the-art methods suggest splitting the RS images and extract small patches suitable for training the DN model \cite{technical_tutorial}.

This methodology (which will be further explained in section \ref{sec:Standard architecture} and a visual representation can be found in Figure \ref{fig:Standard architecture}) is effective for training the DN. But while performing predictions in a complete RS image the process of cropping, predicting and merging the predicted patches, introduces two issues. 

The first problem is the multiplication of image borders during the splitting process. DN models for semantic segmentation, such as UNet++\cite{unet++} and DeeplabV3\cite{deeplab} that will be used in this research, are made up primarily of multiple convolutional layers. In those layers, a kernel of given size “slides” over the 2D input data (a patch), performing an element-wise multiplication. The multiplication results are then summed up into a single output pixel. The kernel will perform the same operation for every location it slides over, transforming the input 2D matrix of features into a different 2D matrix of features.\cite{CNN_guide}.

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/Padding.png}
  \caption{a) Representation of the functioning of a convolutional layer without padding where the kernel of dimension 3$\times$3 sweeps over the layer's input within its borders. The output of the convolution has a smaller dimension with respect to the input. 
  b) Functioning of a convolutional layer with the use of 0-padding. The kernel of dimension 3x3 sweeps over the layer's input moving over the borders such that the dimension of the convolution's output equals the input's one \cite{CNN_guide}.}
  \label{fig:Padding}
\end{figure}


As shown in Figure \ref{fig:Padding} if the kernel has dimensions wider than 1$\times$1, is required the application of a zero-padding method (or similar approaches) before multiplying the kernel and the feature map on the pixels laying on the border of the image. 

Irrespective of how ingenious the use of methodologies such as zero-padding, the predictions that a model performs along the edges of an image are less accurate than the ones performed on the centre of the image. As a matter of fact, while those last are based purely on the overlapping of the kernel on an authentic matrix being a portion of the input, the ones on the edges are the result the products between the kernel and matrixes that are partially coming from the input and partially made up. 
Since the  RS images, have to be split for performing the semantic segmentation, there is an increase in the number of borders, and therefore a propagation of the number of the predictions carried out on partially made up matrixes. In particular, it can be identified a place of points (pixels) in an image, being a grid with the distance between the lines equal to the dimension of the patches cut out from the image; where the described accuracy decrease occurs. 

To deal with this issue In chapter \ref{subsec: Staggering cropping} we propose an architecture aiming to make the predictions less dependent on the way the original RS images are cropped, and in chapter \ref{subsec:Staggering cropping reuslts} we show the performance of that architecture. 

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/context.jpeg}
  \caption{a) Semantic segmentation application whereby the single image the features to be classified are entirely included.
b) Semantic segmentation applied to an RS image. It is made explicit how the limitedness of the information included in a single patch (in the left)  makes hardly recognisable the morphological feature captured from the picture, while with a broader outlook, also from a human point of view, the LCC is more obvious. }
  \label{fig:context}
\end{figure}
The second problem introduced by the process of cropping, predicting and merging the prediction results is the incompleteness of the information included in a single patch. And having patches including incomplete (cropped) objects makes the segmentation tougher. 
In typical segmentation applications (example in Figure \ref{fig:context}), most of the features that have to be classified are completely included within the boundaries of the image. 

In RS images the classification target: morphological features of the land cover, can have highly variable dimensions. A woodland, for example, can be a wood covering a small area as a forest of hundreds of thousands of square kilometres. With the dimension of a patch being 512$\times$512 that with a 10-meter ground resolution leads to an area of 25 thousand square kilometres. That area could be limited in a few conditions and not be sufficient to fit an entire morphological feature in a single patch. Thus, a DN based classification over RS images can't rely on the borders of a feature, but must be based only on the colours of the pixels and patterns included on the single patches. 
A trivial solution would be to downscale the RS image before cropping to obtain patches less detailed but covering wider areas, that would however lead to prediction with lower resolution and would imply an intolerable loss of information. 

To face this issue in chapter \ref{subsec:Multi-Stage fusion} we propose an architecture that combines the level of detail of the predictions performed on RS images at the original dimension with predictions performed on patches coming from downscaled RS images yielding wider contextual features. 
Architecture whose performances is tested in section \ref{subsec:Multi-Stage fusion results}.

This research aims to estimate the feasibility of those two architectures meant to solve the previously exposed problems and evaluate their performances. 



\section{Related Works}
\label{sec:Related Works}
\subsection{Land cover classification}

% which are the techniques
With the improvement of spatial resolution of remote sensing (RS) images, RS image classification gradually formed three parallel classification branches at different levels: pixel-level, object-level, and scene-level classification \cite{LCC-pixel-object-scene} whose examples are shown in Figure \ref{fig:classificaiton_levels}.
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/classification_levels.jpeg}
  \caption{Three level of remote sensing classification. (a) Pixel-level: labels each pixel with a class. (b) Object-level: recognises meaningful semantic entities in remote sensing images. (c) Scene-level: classifies each given remote sensing image patch into a semantic class \cite{LCC-pixel-object-scene}.}
  \label{fig:classificaiton_levels}
\end{figure}

The pixel-level classification task consists in the labelling of each pixel in the remote sensing images with a semantic class. Essentially the pixel-level classification in the remote sensing field corresponds to the segmentation tasks in the computer vision word. 
This branch was the central topic of the early LCC literature. Researchers mainly focused on classifying remote sensing images at pixel-level \cite{early_works} since the spatial resolution of remote sensing images was very low and the size of a pixel was similar to the sizes of the objects of interest. With the increase of the level of detail of remote sensing images captured by satellites, the scientific community interest moved also on the possibility of identifying smaller and more specific objects in the ground (object-level classification) and merging the information linked with those objects to attribute semantic classes to remote sensing images.

% what offers pixel level classification 
Nevertheless, the pixel-level classification remained a subject of great interest for the researchers since it allows to monitor the landscape change detection and permit land use/cover classification. Thus, it will be the kind of LCC on which this research focuses.

\subsection{Remote sensing segmentation techniques}
% breve intro che espanderò nella tesi 
RF and SVM are the most-common classic classifiers used in literature for the land cover classification of remote sensing data. Like the other non-parametric supervised classifiers, these algorithms do not make any assumption regarding the distribution of data and they have shown promising results in classifying remote sensing data overtaking the field’s earlier classifiers adopted such as Linear Regression (LG), Maximum Likelihood (MLC), K Nearest Neighbor (KNN) and Classification and Regression Tree (CART)\cite{PoliMi-review-LULC}. However, in the last few years, deep learning models such as CNNs have been proved to be very successful in classifying complex contextual images, and have been widely used to classify remote sensing data, outperforming the classic methods\cite{DNN-LULC}.
Between those, U-Net++ and DeeplabV3+ stand out for performance. Moreover, those two are simple to implement even with modest resources because their weights  pre-trained on ImageNet \cite{imagenet} are readily available;   this reduces significantly the successive training time on RS images.  Thus, they have been used as fundamental blocks in our new architecture presented in section
\ref{subsec:Multi-Stage fusion}.

\subsection{U-Net++}

Unet \cite{unet_lcc} in Figure \ref{fig:unet} is a symmetric architecture consisting  of two major parts. The left part is called the contracting path, which is constituted by the usual convolutional process: a sequence of down-sampling blocks, each consisting of a sequence of 2 convolutional layers with a 3$\times$3 kernel size and a padding of 1 pixel. The first element of each block results from the application of the max-pooling layer (with a 2$\times$2 kernel and stride of 2) on the result of the previous block. The right part is expansive path, which is constituted by transposed bidimensional convolutional layers performing an up-sampling on their respective input. The interesting feature of UNet is the input of the up-sampling blocks, which is not made only from the result of the previous block, but also from the down-sampling blocks on the left side having the corresponding output dimension.  Those “skip connections” allow the contextual information typical of less-processed images to flow into the convolutional layers without losing the localization accuracy caused by the down-sampling of the max-pooling steps.
UNet++, proposed by Zhou et al. in 2018 \cite{unet++}, is an evolution of UNet succeeding to improve the accuracy of the previous architecture by adding more intermediate (nested) convolutional blocks and densifying the skip connections between blocks.  Following the thesis of Zhou, having a denser structure implies an increased semantic similarity between the results of each couple of concatenated up-sampling blocks, resulting in an easier optimization problem to solve in the training and therefore an higher overall accuracy. 
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/unet.png}
  \caption{UNet architecture. The input multi-channel feature map passes through a sequence of layers in an encoder-decoder structure. The lines connecting the layers illustrate how the information flows through different blocks of the architecture and the layer's colour denote the different mathematical operations carried out.}
  \label{fig:unet}
\end{figure} 



\subsection{DeeplabV3+}
\label{subsec:deeplab}
The Deeplab architecture \cite{deeplab}, as well as its successive evolutions, has been proposed by Liang-Chieh Chen and the Google team\cite{deeplab}.  To date, four versions have been launched. The last release of the network, DeeplabV3+, likewise UNet has an encoder-decoder structure with the encoder being based on Xception \cite{Xception} with a number of variations proposed by Liang-Chien Chen. In terms of performance, two of the most impactful additions to were the Atrous Convolution (AC) and Spatial Pyramid Pooling (SPP). AC allows to include on the prediction a larger receptive field while SPP makes the model capable of carrying out equally accurate predictions on features of different scale. 

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/Atrous.png}
  \caption{Representation of an Atrous Convolution of kernel 3$\times$3 and stride 1. Although having a 3$\times$3 kernel the convolution covers an area on the input image of 5$\times$5 pixels guaranteeing a better receptive field.}
  \label{fig:atrous}
\end{figure}

In more detail, the Atrous convolution (in Figure \ref{fig:atrous}) is a convolution where the kernel is implemented with an internal spacing (stride) given by a parameter "rate". With AC the kernel samples the input values every rate pixels in the height and width dimensions. Thus it receives as input a larger receptive field without letting the resolution decrease. Moreover, AC is computationally lighter than the standard convolution and makes the models which are making use of it way faster. 

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/SPP.jpeg}
  \caption{The SPP divides the feature map, outoput of the convolutional layers, into n x n bins (here n being equal to 4, 16, 64). The bins are then collected in a fixed length representation vector, which is given as input to the fully connected layer. This block allows the model to learn features of different scales. \cite{SPP}}
  \label{fig:SPP}
\end{figure}

The Spatial Pyramid Pooling (SPP), which was first proposed by He et al \cite{SPP}, is a building block typical of the Deeplab architectures, which substitutes the single pooling layer commonly used at the end of a NN between the last convolutional layer and the fully connected layer. 
SPP resamples the given feature layer at multiple rates prior to convolution. Thus, the feature map is divided in a fixed number of spatial bins having sizes proportional to the image size. Each bin gives a different scaled image (see Figure \ref{fig:SPP}). The bins are eventually flattened and given as input the fully connected layer. This, in contrast with the AC, adds learning complexity and slows down the training process, but makes the model capable of coping with features of different scales.  

Clearly, both models were designed with the aim of improving the receptive field and thus, to make the resulting predictions more dependent on the input images as a whole. 
This research intends to follow this same principle, applied to the domain of remote sensing, and proposes a solution to extend it with the parallelization of more models performing predictions at different levels of ground resolution.
 

\section{Data}
\label{sec:method}

The North Holland province, with the website "Copernicus Open Access Hub" \cite{copernicus_open_access_hub}, has access to RS images collected from the ESA's\footnote{ESA: European Space Agency} Copernicus Sentinel-2 mission. From those images, coming in different formats and bandwidths, the more appropriate for the LCC problem are the ones having a higher resolution (10 meters per pixel) and capturing all the visible features of the land cover. 

The hub, considering the short revisit time of the Sentinel's satellites\cite{satellite_description} and that the mission started making the data public in December 2016, provides a huge amount of those images. 
Nevertheless, those data are \textbf{\emph{unlabelled}}, and to train the model a ground truth is needed. Therefore it has been necessary to look for an external labelled dataset of RS images similar to the Sentinel's. Most of the available annotated datasets were inadequate to fulfil the task though. E.g. the Sentinel-2 level 1c classification, the most straight forward choice since is performed over the Sentinel-2 data itself \cite{sentinel_2_classification}, has only one generic label to describe the vegetation. Thus, a model trained on those data cannot distinguish between different types of vegetation and is therefore inadequate for the province purpose of monitoring the mutations of the forest.  

Moreover the research aims to address the issue using Deep Networks\cite{DNN-LULC}, and those require large labelled datasets to be successfully trained and reach high accuracies.

A dataset that is sufficiently large and shares a high degree of similarity to the Sentinel2 is the GID dataset \cite{GID2020}. 

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/GID_data.png}
  \caption{example of an fine land cover classification annotation from the GID dataset  \cite{GID2020url}. On the left the original satellite image, on the right a representation where each class is paired with a distinctive colour.}
  \label{fig:GID_data_exe}
\end{figure}

The GID dataset is a \textbf{\emph{labelled}} dataset collected from the Chinese satellite Gaofen-2. Despite having different construction, and operating in a different way to the satellites of the Sentinel-2 mission, GF-2 gathers datas that are sufficiently similar to those of the Copernicus mission. 

While creating the GID \footnote{GID: Gaofen Image Dataset} dataset Tong et al. \cite{GID2020} produced two different types of pixel-level annotation: a large-scale classification set and a fine land-cover classification set (example in Figure \ref{fig:GID_data_exe}). For the large-scale, they annotated 150 GF-2, and for the fine LCC\footnote{LCC: Land Cover Classification} set they labelled 10 GF-2.  GID is widely distributed over geographic areas covering more than 70,000 $km^2$. Benefiting from the various acquisition locations and times, GID presents rich diversity in spectral response and morphological structure. For the large-scale dataset five representative land cover categories are selected to be annotated: built-up, farmland, forest, meadow, and waters. Areas that do not belong to the above five categories are labelled as unknown (or background). On the other hand, albeit having a smaller amount of annotated data the fine LCC set comprises 15 mask categories: industrial land, urban residential, rural residential, traffic land, paddy field, irrigated land, dry cropland, garden land, arbour forest, shrubland, natural meadow, artificial meadow, river, lake and pond. 

In the training process of architectures we proposed, loss of information is intrinsic because of the required  down-scaling of the satellite images (details will be explained in chapter \ref{Architectures}). Therefore, in order to have a sufficient amount of data to guarantee a proper training of the DN, it has been chosen the large scale GID dataset.

\section{Proposed Architectures}
\label{Architectures}
\subsection{Standard architecture}
\label{subsec:Standard architecture}
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/Standard_Architecture.png}
  \caption{Illustration of the Standard Architecture inspired from the work of Liu et al. \cite{Patches}. The remote sensing images are splits into several small patches and each of them is fed into the deep convolutional network. The predictions are eventually merged into one single output. }
  \label{fig:Standard architecture}
\end{figure}
As introduced in section \ref{sec:intro}, to perform the semantic segmentation on remote sensing images through the use of deep learning models, it is needed to preprocess the input of the model and split it into small patches.
In Figure \ref{fig:Standard architecture} is illustrated the common way the input is processed before the prediction and merged afterwards. A similar architecture has been proposed by Liu et al\cite{Patches}, whose also proposed a partial overlapping of the patches. Here we describe an even simpler version, without overlapping, which will be used as benchmark for the performances of the other architectures we proposed and analysed. 

The satellite images of dimensions 7168$\times$6656$\times$3 (width, height, channels) are split into patches of dimension 512$\times$512$\times$3.

The patches are given as input to a model chosen as a baseline, either UNET++ or Deeplabv3 as explained in the related works section. Which is pre-trained on IMAGENET and further trained on patches of the same dimension coming from the GID dataset. From now on this trained model will be called M1.

M1 acts by mapping each pixel of the input patch with a probability vector stating the probability with which it belongs to one of the 6 classes the GID dataset is labelled with.  Therefore, the output is a matrix of dimension 512$\times$512$\times$6, with the dimensions being respectively width, height and class.

The model's outputs are collected and merged back into a sole prediction matrix with dimensions (7168$\times$6656$\times$6) coherent with the original dimension of the RS image.

Eventually, applying the Argmax function to the prediction matrix, along the class axis, is obtained a bidimensional matrix of dimension (7168$\times$6656) having in each pixel a number corresponding to the class it belongs to.

This standard approach, though, introduces two challenges:
\begin{itemize}
    \item The predictions performed on the borders of the patches are less accurate than the ones done in the centre of the patches.
     \item It's hard to reach high accuracies because of the loss of context features while focusing the predictions on small areas. 
\end{itemize}
 

\subsection{Staggering cropping}
\label{subsec: Staggering cropping}

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/Staggering_Cropping.jpeg}
  \caption{Illustration of the Staggering architecture we proposed and described in section \ref{subsec: Staggering cropping}.}
  \label{fig:Arch_staggering}
\end{figure}

As explained in the Introduction, the models rich in convolutional layers such as DN models have a slight performance decrease while making predictions along the borders of the matrixes given to them as input. And while splitting an image in a large number of patches there is a propagation of this issue. Here, we propose an architecture aiming to prevent possible inaccuracies decoupling the splitting process from prediction outcome.

The architecture, illustrated in Figure \ref{fig:Arch_staggering}, consists of two prediction pipelines paired in parallel. 

The first pipeline of the architecture is equal to the standard architecture except for the absence of the Argmax function.
Then, the input is an RS image of dimension (7168$\times$6656$\times$3) and the output is a matrix of probability vectors of dimension (7168$\times$6656$\times$6).

The second pipeline, on the other hand, has a few differences from the standard one. 

The input is taken cropping the original input RS image such that is left an offset of 256 pixels from all the borders of the image. Therefore is obtained an input image of dimension (6656$\times$6144$\times$3).


The image is then split in patches of dimension (512$\times$512$\times$3). Thanks to the previous step, the patches obtained don't have any border in common with the patches got from the first pipeline.

The patches are then processed from M1.

The model's outputs are collected and merged back into a sole prediction matrix with dimensions (6656$\times$6144$\times$6).

Eventually all the borders of the prediction matrix are zero-padded to such that is obtained a matrix of dimensions (7168$\times$6656$\times$6) coherent with the input.

Eventually, the mean is taken of the probability vectors matrices of the outputs of the two parallel pipelines. 
The mean is then given as input to an Argmax function, applied along the class axis, leading to a bidimensional matrix of dimension (7168$\times$6656) as overall output.


\subsection{Multi-Stage fusion architecture}
\label{subsec:Multi-Stage fusion}
\begin{figure*}[ht]
\centering
  \includegraphics[width=\linewidth]{Images/Multi_stage_Architecture.png}
  \caption{Illustration of the Multi-Stage fusion architecture we proposed and described in chapter \ref{subsec:Multi-Stage fusion}.}
  \label{fig:Multi_Stage}
\end{figure*}

To deal with the issue of the loss of context features resulting from the splitting in patches of the RS images is proposed a second architecture (in Figure \ref{fig:Multi_Stage}). Its purpose is to increase the overall prediction accuracy avoiding the trade-off between predictions on images with higher resolution and images less defined but covering a wider area on the ground.

The architecture can be divided into three pipelines. The first one is exactly similar to the standard architecture with only the difference being that the output P1 is not processed from the Argmax function.

The second and the third pipeline, on the other hand, are similar to the standard architecture, but they are processing different data.

Respectively in the second and in the third pipeline the original images of size (7168$\times$6656$\times$3) are downsized with scale to size 2 to (3584$\times$3328$\times$3) and with scale 3 to (2389$\times$2218$\times$3).

The downsized images are split in patches of dimension (512, 512, 3).
The patches of the second stage will have a ground resolution of 20 meters and a covered area of 1024$\times$1024 $km^2$ and the patches of the third stage will have a ground resolution of 30 meters and a covered area of 1536$\times$1536 $km^2$.

The patches are given as input to the models M2 and M3. These models are trained with the training set portion of the GID dataset, where the images have also been downscaled accordingly to the expected dimension of the input of the models and split into patches of dimension (512,512,3). The downscaling is performed with the nearest neighbour as interpolation criteria.

The outputs of the models are a collection of patches that, as in the first stage, are merged. Therefore, stage 2 leads to a matrix of prediction of dimension (3584$\times$3328$\times$6) and stage 3 to a corresponding matrix of dimension (2389$\times$2218$\times$6).

Both those matrixes are upscaled, in the respective pipelines, to matrixes P2 and P3 of dimensions equal to the first stage output.


Eventually, the outputs are leading to three distinct matrices of prediction vectors resulting from three models trained with different levels of ground resolution and covered area. 

That information is joined by summing the matrices weighted with three different weights w1,w2 and w3, one for each stage. 

$$P=w1\cdot P1 + w2\cdot P2 + w3\cdot P3$$

An Argmax function (applied on the class axes) is then fed with the matrix M and returns an overall output being a bidimensional matrix of dimension (7168$\times$6656) having for each pixel the number relating to the class to which it belongs.


\section{Results and Discussion}

\subsection{Dataset splitting}
Before proceeding with the training of the M1, M2 and M3 the dataset has been split into three sets: training, validation and test having a respective fraction with the overall GID dataset of 80\%, 10\% and 10\%. 
\begin{itemize}
    \item The training set has been used to train the models.
    \item The validation set to measure first the performance of each algorithm and then to tune the weights [w1, w2, w3] of the fusion of different stages, and also to assess the utility of the staggering architecture. 
    \item The test set was eventually needed to return an unbiased metric of the overall performance of the architecture. 
\end{itemize} 

\subsection{Data Processing}
\label{Data Processing}

The GID dataset is distributed from its creators as a directory containing two folders: one with the RS images and one with the corresponding labelled images. Those last are images with three channels (RGB) because each label is represented as a colour:
\begin{itemize}
	\item{black: not classified}
	\item{red: built-up land}
	\item{green: farmland}
	\item{light blue: forest}
	\item{yellow: meadow}
	\item{blue: water}
\end{itemize}
To make the dataset lighter the labelled images are label encoded, meaning that the 3 channels are mapped into a single channel with values integers and constrained between 0 and 5: one for each possible class.

Then the GID dataset is further processed such that a new directory is created for each level of ground resolution associated with the stages of the Multi-Stage fusion architecture. 

The first directory D1, thus, will contain two folders, one having the 512$\times$512$\times$3 patches obtained splitting the images having the original ground resolution, and the other one including the correspondent labelled patches of dimension  512$\times$512 obtained cropping the one label encoded ground truth data. 
The second and the third directories (D2 and D3) are made of patches of the same size taken after downscaling the RS images and the labelled images with a respective dimension of 1/2 and 1/3 compared with the original GID data.

\subsection{Implementation details}
Each of the Multi-Stage fusion architecture’s stages requires a semantic segmentation model trained on a dataset of patches of the stage-specific ground resolution. Therefore, the model performing predictions on the first stage, M1, is trained on patches with a ground resolution of 10m, the training set of D1. Similarly, the models M2 and M3 are trained respectively on patches of ground resolution 20m and 30m, being the training sets of D2 and D3.

The architecture of the models M1, M2 and M3 is Unet++ or DeeplabV3+ (both are tested). Regardless of the type of base architecture, the models are pre-trained on ImageNet \cite{imagenet} to speed up the training process.
Moreover, to make the training processes of the two models comparable, both of the architecture have been trained with a batch size of 8 and using as optimizer Adam, which is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. 
Adam requires the following parameters as inputs:
\begin{itemize}
    \item $\alpha$: Step size parameter (set to 0.001)
    \item $\beta_1$: Used for decaying the running average of the gradient (set to 0.9)
    \item $\beta_2$: Used for decaying the running average of the square of gradient (set to 0.999)
    \item $\epsilon$: Meant to prevent Division from zero error. ( set to $10^{-8}$)
\end{itemize}
Driven to the same purpose of confronting the results of the training of each model (M1, M2 and M3 for based on either DeeplabV3+ or Unet++), they have been trained for a fixed time of 24 hours on an NVIDIA T4 GPU of Google Colab pro.

\subsection{Metrics}
The objective while training the models M1, M2 and M3 was to minimise the Dice Loss, or rather to maximise the Dice coefficient. Which is metric commonly used in semantic segmentation applications to judge the bounty of the pixel-by-pixel predictions. 
The Dice coefficient (in Figure \ref{fig:Dice_loss}) measures the overlap between the masks labelled from the model and the ground truth's ones.
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/dice_loss.jpeg}
  \caption{Dice Loss}
  \label{fig:Dice_loss}
\end{figure}

Each class $i$ of the GID dataset \ref{Data Processing}, except for the background class, is then evaluated with the $Dice\:loss_i $ and the number of pixels $p_i$ of the ground truth belonging to that class.
Then, as overall accuracy metric is calculated the $multiclass\:Dice\:loss$: the weighted average of the Dice losses of every single class.
\begin{equation}
    Multiclass\:Dice\:loss\:=\:\frac{\sum{Dice\:loss_i}*p_i}{\sum{p_i}}
\end{equation}

\subsection{Standard architecture}
\label{subsec:Standard architecture results}  
In order to give a benchmark to the result, we tested the models M1, M2 and M3 with an architecture commonly used for using deep learning models in the remote sensing field, such as the one described in chapter \ref{subsec:Standard architecture}
Thus, we tested the models on validation. In the following table are shown the resulting Multiclass Dice coefficients for each possible combination of stages of the Multi-Stage fusion architecture (thus ground resolution of the data) and type of model the architecture is based on.

\begin{center}
\label{tab: single model results}
\begin{tabular}{ |c|c|c|c| } 
 \hline
  & stage1 & stage2 & stage3 \\ 
 \hline
 UNet++ & 79.4\% & 77.4\% & 71.1\% \\ 
 \hline
 DeeplabV3+ & 82.0\% & 78.1\% & 77.5\% \\ 
 \hline
\end{tabular}
\end{center}

\textbf{\\Table 1: Standard architecture multi-class Dice coefficients for each combination of models - levels of ground resolution}
\\

With similar hardware and fixed the training time, DeeplabV3+ reaches higher accuracies than Unet++ for each level of ground resolution of the training set. 

Those results make clear clear how the scarcity of data is a fundamental problem while dealing with the land cover classification task. 
In general, it is challenging for the model to learn a variety of morphologic features of the land cover from a dataset of 150 images even though they are covering a wide area on the ground. This is particularly clear for the models M2 and M3 which are suffering from underfitting. 
This is because the training set is composed of 120 images, which for the second and the third stage of the multi-layer architecture are down-scaled to a lower resolution. The down-scaling leads to a smaller number of patches per image, indeed, while M1 is trained on 120*182 patches, M2 is trained on 120*46 patches and M3 is trained on 120*20 patches. 
Thus, another piece of evidence carrying to this observation of underfitting is that, independently from the chosen architecture (UNet++ or DeeplabV3+), the accuracy of M1 [79.4\%, 82.0\%] is way higher than M2's [76.4\%, 78.1\%], which is way higher than M3's [71.1\%, 77.5\%]. Such a variation would not have been as large if it was not for the dependency of the performance on the dimension of the training set. 

This statement could appear in contrast with the general, but still moderate, loss of performance observed while experimenting with the architecture on the test set. Nevertheless, this seeming contradiction finds an explanation again in the scarcity of the data available. 
Indeed, as we split the GID dataset, made of 150 RS images, with an 80\%, 10\%, 10\% proportion (respectively training, validation and test set), the test set is limited to only 15 images. The patches inside each image are often strongly correlated, therefore if there are one or a few images inside of the test set coming from previously unseen landforms which are badly classified from the models they will have a heavy influence on the average Dice coefficient.
Those observations suggest that for further analysis of this architecture either to start from a bigger dataset or at least to exclude from the training and testing process those RS images which differ too much from the ones the architecture will be supposed to classify while used in. a practical application. 

Another consideration coming from the observation of the results for the single models in the context of the standard architecture, is the effectiveness of the peculiar building blocks of DeeplabV3+ in the LCC application. Indeed, thanks to SPP and the atrous convolution (explained in chapter \ref{subsec:deeplab}), DeeplabV3+ is particularly suited for images containing features of different scale and outperforms Unet++ for each ground resolution of the training set. 

\subsection{Staggering cropping results}
\label{subsec:Staggering cropping reuslts}
After analysing the Standard architecture accuracy which provides a performance benchmark for the other proposed architectures, we studied the staggering cropping. The model M1 is used as a fundamental block for performing the predictions in the two parallel pipelines of the staggering architecture. 
The latter is then tested on the validation set and the results are compared with the ones of the so-called standard architecture.
\begin{center}
\label{tab: Staggering Cropping}
\begin{tabular}{ |c|c|c| } 
 \hline
  & Standard Architecture & Staggering Cropping \\ 
 \hline
 UNet++ & 79.4\% & 79.0\% \\ 
 \hline
 DeeplabV3+ & 82.0\% & 82.0\% \\ 
 \hline
\end{tabular}
\end{center}
\textbf{Table 2: Comparison of the multi-class Dice coefficients between the Standard architecture and Staggering cropping architecture with the model M1 being either UNet++ or DeepLabV3+}
\\ 

The results in table \ref{tab: Staggering Cropping} show that the addition of a second prediction pipeline in parallel to the standard one with the aim of decoupling the labelling performed by the model from the way the original RS images are cropped, did not yield the desired improvements, leaving the Dice loss almost unchanged. The results prove the ineffectiveness of the staggering architecture. 

This outcome can be explained by observing that the proposed solution mainly influenced the predictions performed along the patches' borders. Even considering the border-pixels in an image processed in the standard architecture from the M1 model, the one requiring the input image to be split in the higher number of patches, their percentage is negligible over the total number of pixels of the image. Going into the specific numbers, each patch has 2044 border pixels, which, multiplied by the number of patches per full resolution RS image from the GID dataset, 182, means a total of 372008 pixels that have to be labelled from the model through the help of 0-padding. That includes slightly less than 1\% of the total pixels of the RS image. 

Moreover, the architecture is based on two models having convolutional layers with small kernels. Unet++ has convolution layers with 3$\times$3 kernel and Deeplab has atrous convolution layers kernel of stride 2 and dimension 3$\times$3. This means that 0-padding cannot have contributed for more than $\frac{1}{3}$ to the predictions performed on the borders, having thus only a minor influence on the model result. 

\subsection{Multi-Stage fusion results}
\label{subsec:Multi-Stage fusion results}

Eventually, also the Multi-Stage fusion architecture is examined. Its performances are evaluated both in a two-stage architecture, including only stage one and stage two, and in its complete three-stage shape. This, to determine how necessary such a complex design is, or whether a two-stage architecture would instead be sufficient. 

For the two-stage configuration, the architecture is tested on the validation set of the GID dataset for each possible permutation of the models UNet++ and DeeplabV3+, and for every possible combination the weights w1 and w2, taken as integers and bounded in a range within 1 and 9. In the second column of the table is highlighted the couple of parameters [w1, w2] leading to the higher accuracy for the final outcome of the architecture.

\begin{center}
\label{tab: single model results}
\begin{tabular}{ |m{9em}|c|c| } 
 \hline
 Model's combination & [w1,w2] & overall accuracy \\ 
 \hline
 Stage\_1:UNet++ Stage\_2:UNet++ & [4,1] & 79.7\% \\ 
 \hline  
 Stage\_1:UNet++ Stage\_2:DeeplabV3+ & [5,1] & 79.6\% \\ 
 \hline
 Stage\_1:DeeplabV3+ Stage\_2:UNet++ & [5,1] & 82.2\% \\ 
 \hline
 Stage\_1:DeeplabV3+ Stage\_2:DeeplabV3+ & [7,3] & 82.4\% \\ 
 \hline
\end{tabular}
\end{center}
\textbf{\\Table 3: Two Stages Fusion architecture multi-class Dice coefficients. Which are evaluated, on validation set, for each combination of models and for the pair of parameters [w1, w2] maximising it.}
\\

For the complete design with three stages, on the other hand, for sake of simplicity, the architecture is tested only with each model (M1, M2, M3) being Unet++ or each model being DeeplabV3+. Again, the second columns is showing the combination of weights[w1, w2, w3] that while used on the fusion block are leading to the best multiclass Dice loss, and the third column is contains the fixed $\vec{w}$ the overall accuracy on validation.

\begin{center}
\label{tab: single model results}
\begin{tabular}{ |m{9em}|c|c| } 
 \hline
 Model's combination & [w1,w2,w3] & overall accuracy \\ 
 \hline
 Stage\_1:UNet++ Stage\_2:UNet++ Stage\_3:UNet++ & [9,8,8] & 80.1\% \\ 
 \hline
 Stage\_1:DeeplabV3+ Stage\_2:DeeplabV3+ Stage\_3:DeeplabV3+ & [9,2,3] & 82.4\% \\ 
 \hline
\end{tabular}
\end{center}
\textbf{\\Table 4: Three Stages Fusion architecture multi-class Dice coefficients. Which are evaluated, on validation set, for each combination of models and for the pair of parameters [w1, w2] maximising it.}
\\

In he results in table 3 and table 4 the increase of Dice Coefficient supports the idea of classifying an RS image merging predictions performed on patches with different levels of ground resolution and area coverage.
Moreover, the experiments show how the improvement brought by Multi-Stage fusion architecture is more pronounced when UNet++ is used for each stage of the architecture. On the other hand, while using DeeplabV3+ the improvement added from the architecture partially overlaps with the effects of the atrous convolution and the spatial pyramid pooling, which already enlarge the receptive field of the model making it capable of performing prediction at very different scales. 

Going into the details of the 3-stage architecture, the results suggest that the small improvement of the overall result, 0.4\% for UNet++ and less than 0.1\% for DeeplabV3+, does not justify the increase of complexity linked to adding a third parallel pipeline in the architecture and training a third model. 
This poor result is also due, as previously explained in the discussion over the single models' performances, by the scarcity of data involved in the training of M3. M3 strongly suffers  of underfitting and is not comparable with M1 and M2, it introduces in the fusion layer at the end of the architecture more noise than added value. 

Because of this observation, it has been taken into account only the two-stage architecture for an in-depth analysis of the contribution of the predictions carried out in each single stage to the overall accuracy of the outcome. The analysis has been performed summing with different weights the outputs of single models in the parallel pipelines of the architecture, and evaluating for each couple of weights [w1, w2] the overall accuracy of he architecture. The result is presented in Figure \ref{fig:result weights} as a color map.


\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/DD_on_val.jpeg}
  \caption{Colour map showing on the y-axis the contribution w1 (thus of the first stage predictions) and on the x-axis the contribution w2 (thus of the second stage predictions) in the overall Dice coefficient. The legend on the right shows that the darker the colour of the cell in the matrix, the higher the multi-class Dice coefficient of the model's output for that combination of hyperparameters [w1, w2]. In particular, this colour map analyses the multi-class Dice loss on the validation set when DeepLabV3+ is used as the basic model of the Multi-Stage architecture for each stage. In addition in the map is highlighted the line related ideal ratio $w1/w2$ maximising the output's accuracy.}
  \label{fig:result weights}
\end{figure}

The resulting matrix shows that even if the multi-class Dice loss evaluated on stage-1 outnumbers the one calculated on stage-2, the combination of results with different levels of ground resolution carried out by the Multi-Stage fusion architecture improves the overall accuracy of the land cover classification task. 
Moreover, it shows the combination of the parameters 

$$[w1,w2]_{best}\ |\ DiceLoss(P) = DiceLoss(\vec{w} \cdot \vec{P})\ is\ minimized$$ 

To prove the consistency within the results obtained in validation, the Multi-Stage architecture we proposed has been tested on test-set fixing the hyperparameters w1 and w2 to the couple of values $[w1,w2]_{best}$.

\begin{center}
\label{tab: single model results}
\begin{tabular}{ |m{9em}|c|m{2em}|c| } 
 \hline
 Model's combination & [w1,w2] & SMA & overall accuracy \\ 
 \hline
 Stage\_1:UNet++ Stage\_2:UNet++ & $[4,1]_{fixed}$ &78.3\% 72.1\%& 78.4\% \\ 
 \hline  
 Stage\_1:UNet++ Stage\_2:DeeplabV3+ & $[5,1]_{fixed}$ &78.3\% 73.6\%& 78.5\% \\ 
 \hline
 Stage\_1:DeeplabV3+ Stage\_2:UNet++ & $[5,1]_{fixed}$ &78.0\% 72.1\%& 78.4\% \\ 
 \hline
 Stage\_1:DeeplabV3+ Stage\_2:DeeplabV3+ & $[7,3]_{fixed}$ &78.0\% 73.6\%& 78.9\% \\ 
 \hline
\end{tabular}
\end{center}
\textbf{\\Table 5 :Two Stages Fusion architecture multi-class Dice coefficients. Which are evaluated, on test set, for each combination of models and for the best pair of parameters [w1, w2] found on validation.}
\\

The parameter SMA (acronym for single model accuracy) shows the accuracy on test of the single models. The results prove the robustness of the architecture showing that the weights learned on validation are effective also in testing. The same result can be deduced from the colour map in Figure \ref{fig:result weights} where the accuracy gain due to the fusion of stage-1 and stage-2 predictions in testing shows the same pattern as the one found in validation \ref{fig:weights test}.
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Images/DD_on_test.png}
  \caption{Colour map showing the multi-class Dice loss on the test set when DeepLabV3+ is used as the basic model of the Multi-Stage architecture for each stage. }
  \label{fig:weights test}
\end{figure}

In accordance with the observation of under-fitting described in section \ref{subsec:Standard architecture results} and the consequent choice of considering only a two-stage architecture, it comes out how the Multi-Stage architecture shape is strongly dependent on the dataset on which it has been trained. The number of parallel stages has to be chosen with respect to the number of examples in the training set and particularly on how many times it can be down-scaled without losing too much information having a too small number of patches.

For the same reason, with the experiments carried out it is not possible to clearly define what would be the ideal ratio between the weights $w1$ and $w2$ heading to the ideal mix of [low resolution - high receptive field] predictions and [high resolution - low receptive field] predictions, assuring the best overall accuracy. Indeed, the values of w1 and w2 are depending also on the difference of accuracy between M1 and M2, since a higher value of w1 makes the overall predictions closer to the one of the model M1, which will obviously be privileged as it outperforms M2. To make possible a correct and unbiased estimation of w1 and w2 it would be needed to study the behaviour of the architecture with M1 and M2 being (almost) equally accurate.

Overall, even without being able to draw a clear line over the contribution (w1, w2) on the total Dice coefficient of different training stages, it is clear how the fusion of those is concurring to a  better result with respect to the standard architecture, hence justifying the need of approaching the LCC problem with deep networks performing predictions over different scales of the same RS image. Moreover, the ratio between w1 and w2, has been proven to be robust, since when fixed the best one on validation, the same performed comparably well on the test set. 

\section{Conclusion}
\label{sec:Conclusion}

The current work supports the idea that combinations in parallel of simple CNN-based models offer a solution for the issues that arise when using deep networks in the land cover classification tasks.
Unsatisfactory results were observed when staggering-cropping was used, which was aimed to decouple the predictions by the accuracy decrease due to the patches boundaries. On the other hand, using an architecture that puts in parallel algorithms trained on RS images with different ratios of ground resolution level and receptive field has been found to bring modest improvements with respect to approaches with standard deep networks.
This last solution could be studied more in-depth in presence of larger ground truth. In fact, if large enough, the downscaling of the data with the consequent information loss, would not be so decisive to cause M2 and M3 to underfit the data. That could give unbiased insights on the contribution in the fusion of the different model outcomes (P1, P2 and P3) and elucidate more precisely the improvement brought from the Multi-Stage fusion architecture. 


An additional result of importance is the power of the Spacial Pyramid Pooling module, partially capable of facing the issue of receptive field scarcity proper of the RS images patches. SPP, if accompanied with the improvement of GPUs computational power, and thus with the possibility of feeding models with wider patches, is then indexed as a potential solution for including more context features in predictions over RS images carried out by convolutional neural networks. 

Overall, and despite the described limitations, this research, carried out with modest resources, was enough to show the feasibility of following the path of putting in parallel models trained with [low resolution - high receptive field]  and with [high resolution - low receptive field] to deal with the issues carried with the use of CNN models in the RS field.

\bibliographystyle{ACM-Reference-Format}
\bibliography{thesis_bib}



%\section{Appendix}
%\label{sec:appendix}
%
%// put images here
\end{document}
\endinput
