# Meth-AINT -- Locating Oil & Gas Infrastructure from Aerial Images

## Background
Methane is a particularly potent greenhouse gas lurking in the shadows of the climate crisis.  While carbon dioxide is the most prominent greenhouse gas, methane gas contributes to 25% of the global warming we experience today and its immediate impact on the environment is 84-fold greater than carbon dioxide over the course of 20-years. Recently, researchers [have exposed](https://www.iea.org/reports/global-methane-tracker-2022/overview) the startlingly 1,800 methane gas leaks between 2019 and 2020 based on satellite imagery of the plumes, largely due to oil and gas refineries, and global methane emissions from the energy sector are 70% greater than reported by national governments. While current satellites can only detect 10% of all methane, [new satellites, such as MethaneSAT](https://www.theguardian.com/environment/2022/mar/06/how-satellites-may-hold-the-key-to-the-methane-crisis) (initiated by the Environmental Defense Fund), will have sufficient resolution to detect methane emissions throughout the world and are expected to launch next year. Identifying methane pollution via satellite is only half the solution, as the plumes must then be linked to the most probable sources of the leak at groundlevel. While satellite images do not have high enough resolution to identify oil refineries and petroleum terminals, the U.S. National Agriculture Imagery Program’s (NAIP) aerial imagery is well-suited for this purpose. Hence, [Stanford’s ML group](https://stanfordmlgroup.github.io/projects/ognet/) has taken advantage of the availability of the NAIP database and curated a high-fidelity dataset of images for this use case.

Surprisingly, oil and gas sectors are incentivized to improve efficiency in their processes. The International Energy Agency predicts that methane leaks in 2021 from fossil fuel operations, if captured and marketed, would have made an [additional 180 billion cubic meters of gas available for sale](https://www.iea.org/reports/global-methane-tracker-2022/overview) (enough gas to power Europe’s energy sector). Thereby having the dual positive effect of mitigating global warming and reducing the severity of the current energy crisis. Methane release from oil and gas infrastructure can be easily abated with practices such as leak detection systems, replacement of pumps, instrument air systems, devices, and compressor seals/rods, as well as installation of flares and plungers. The feasibility of these measures makes this a cost-effective solution for oil and gas companies to cut at least 50% of emissions (ref) by 2030, bringing us significantly closer to the target of 30% reduction in methane proposed by COP26 last year. And at the last COP27 in November, the U.S. Environmental Protection Agency (EPA) announced [more stringent regulations on methane gas emissions by oil and gas companies](https://insideclimatenews.org/news/11112022/new-epa-proposal-to-augment-methane-regulations-would-help-achieve-an-87-reduction-from-the-oil-and-gas-industry-by-2030/). 

## Objective
Locate U.S. oil & gas industry point sources within subset of U.S. aerial images with an optimal recall metric and reasonable precision

## Main Findings
- Severe imbalance requires random oversampling approach (undersampling yielded only minor improvements in metrics)
- DenseNet is most optimal for deep learning of aerial images (as compared to Xception and ResNet)
- Model outcome metrics are sensitive to aggressive augmentation and parameter reduction (via Global Max or Average pooling)


## Dataset
U.S. aerial images were captured between 2015 and 2019 from National Agriculture Imagery Program’s (NAIP) aerial imagery and divided into training, validation, and test sets by the Stanford’s ML group.  Images were taken at 1m resolution and acquired on days with low/no cloud cover, although the research team downsized images to 500 x 500 pixels and reduced resolution to 2.5m. A separate commercial database, Enverus Drillinginfo, was used to identify and label 149 point locations of U.S. oil refineries. Aerial images not containing oil refineries were captured (i.e., “negatives) from U.S. landscapes chosen at random or were targeted “difficult negative” images containing landscapes that are visually similar to the “positive” group. The full dataset contains 7,066 images in total;
- Training set: 127 positive examples, 5,525 negative examples
- Validation set: 13 positive examples, 693 negative examples
- Test set: 9 positive examples, 697 negative examples

## Algorithms
### Feature Engineering
1) Image size was reduced from 500 x 500 to 224 x 224 in order to be processed correctly in CNNs
2) Minority group was randomly oversampled to account for the severe imbalance (2% minority class). Resulting ratio of majority : minority images was 2.4:1 (30% minority class). 
3) Images were then augmented by random flipping, random rotation, and random change in color contrast to increase the variety of images “seen” by the neural net
4) Images were preprocessed using the appropriate CNN built-in preprocessing workflow, which scales the pixel between 0 and 1 and normalizes each channel with respect to the ImageNet dataset

### Models
Models originally tested at baseline were Xception, Resnet50, Resnet152, and DenseNet121 (suggested by the Stanford ML team). 
Model Evaluation and Selection
Ultimately these models did not perform well at baseline (inconsistent recall and precision, generally hovering at 0.00) and therefore several models were tested again using random sampling approaches:
	1)  undersampling of the majority class and 
	2)  oversampling of the minority class. 
Ultimately the oversampling method was chosen with DenseNet121 for final reporting due to superior performance.

## Tools
Tools used include Google Colabs along with python libraries as follows:
- OS/shutil - moving files/folders w/in Colabs
- Pandas, Numpy - for data manipulation
- Matplotlib, seaborn - for data visualization
- Keras (w/in tensorflow)- for running preprocessing, data augmentation and convoluted neural networks
- Sklearn – for model performance metrics 
