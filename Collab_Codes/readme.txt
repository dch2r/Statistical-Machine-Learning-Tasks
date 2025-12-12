-> Task 1: Linear Regression (PyTorch)

Built a linear regression model in PyTorch to predict median housing prices in California.
Performed exploratory data analysis, including distribution and spatial visualizations of features and targets.
Created stratified train/validation/test splits to prevent data leakage and ensure robust evaluation.
Conducted correlation analysis to identify important predictors and guide feature design.
Applied data cleaning and preprocessing, including missing-value handling and imputation strategies.
Implemented extensive feature engineering using pipelines:
One-hot encoding for categorical variables
Feature scaling (min–max and standardization)
Log transformations and ratio-based features
RBF-based similarity features
Trained the model using stochastic gradient descent with a custom training loop.
Evaluated performance using R², error distribution plots, and prediction vs. target visualizations.
Extended the model with 5-fold cross-validation and interpreted learned coefficients.
Added L2 regularization and polynomial basis expansion to study bias–variance trade-offs.
Performed hyperparameter sweeps to analyze the effect of regularization strength on generalization.

-> Task 2: Softmax Regression for Image Classification (CIFAR-10, PyTorch)

Implemented a softmax regression model in PyTorch to classify images from the CIFAR-10 dataset.
Loaded and normalized CIFAR-10 images and created train, validation, and test splits using dataloaders.
Visualized sample training images to understand data distribution and class structure.
Built a sequential PyTorch model consisting of an image flattener followed by a single linear layer.
Implemented the forward pass using softmax and optimized the model with cross-entropy loss.
Trained the model using stochastic gradient descent with configurable learning rate and weight decay.
Ran multiple training experiments with different hyperparameter settings to compare model behavior.
Evaluated model performance on a held-out test set using:
Accuracy and top-K accuracy (K = 1, 2, 3)
Confusion matrices
Classification reports with F1 scores
Analyzed the impact of learning rate and weight decay on classification performance.
Visualized and interpreted learned model coefficients across different trained models.

-> Task 3: Generative Discriminant Analysis and Support Vector Machines (CIFAR-10)

Applied classical machine learning classifiers to the CIFAR-10 image classification task using scikit-learn.
Loaded, normalized, and split CIFAR-10 images into training, validation, and test sets, and visualized sample inputs.
Implemented a Linear Discriminant Analysis (LDA) model using the lsqr solver with covariance estimation.
Trained LDA on flattened image vectors and evaluated performance using:
Accuracy
Confusion matrices
Classification reports with F1 scores
Visualized and interpreted class-wise mean images learned by the LDA model.
Trained Support Vector Machine (SVM) classifiers on a reduced CIFAR-10 subset to handle computational constraints.
Built and evaluated linear-kernel SVMs across multiple regularization strengths (C values).
Selected optimal C values using validation accuracy and analyzed performance trends with semi-log plots.
Trained and evaluated RBF-kernel SVMs using the same hyperparameter selection strategy.
Compared linear and RBF SVM performance on the test set using accuracy and detailed classification metrics.

-> Task 4: Deep Neural Networks and Transfer Learning (CIFAR-10)

Developed a fully connected feedforward neural network and evaluated its performance on the CIFAR-10 image classification task.
Built and trained a convolutional neural network (CNN) from scratch to capture spatial structure in image data.
Experimented with architectural choices and hyperparameter tuning to understand their impact on model performance.
Evaluated and compared fully connected networks and CNNs against linear classifiers such as softmax regression, LDA, and SVMs.
Tested a ResNet-18 model pretrained on ImageNet for out-of-the-box performance on CIFAR-10.
Fine-tuned the pretrained ResNet-18 model on CIFAR-10 and evaluated it on a held-out test set.
Applied data augmentation techniques during fine-tuning to improve generalization.
Analyzed performance differences between models trained from scratch and fine-tuned pretrained networks.
Studied the role of transfer learning and data augmentation when training large-capacity neural networks.

-> Task 5: Transformer-Based Image Captioning (PyTorch)

Implemented a Transformer-based encoder–decoder model in PyTorch for automatic image caption generation.
Built core Transformer components, including attention mechanisms, positional encoding, and sequence modeling.
Integrated visual feature representations with language modeling to generate descriptive captions.
Trained the model on subsets of a large-scale image–caption dataset and analyzed training behavior using loss curves.
Evaluated caption quality and studied the effect of dataset size and training configuration on performance.
Designed the implementation to run efficiently on CPU for smaller datasets, with support for GPU acceleration for larger-scale training.
Organized model code and auxiliary modules to enable modular experimentation and extension.

-> Task 6: Self-Supervised Representation Learning with SimCLR (CIFAR-10)

Implemented the SimCLR self-supervised learning framework to learn image representations from unlabeled CIFAR-10 data.
Designed contrastive learning pipelines with data augmentations to generate positive and negative pairs.
Trained an encoder network to maximize agreement between augmented views of the same image.
Learned feature embeddings without using class labels and analyzed representation quality.
Evaluated learned representations through downstream image classification performance.
Studied the benefits of self-supervised pretraining compared to training models. 

-> Task 7: See kmeans.ipynb file for more detail

-> Task 8: See pca.ipynb file for more detail

-> Task 9: Gaussian Mixture Models for Unsupervised Learning (MNIST)

Studied and implemented a Gaussian Mixture Model (GMM) from scratch using PyTorch.
Applied GMMs to the MNIST dataset to uncover latent structure in high-dimensional image data.
Analyzed learned mixture components and their relationship to digit classes.
Performed model selection using grid search with scikit-learn’s GMM implementation.
Identified optimal hyperparameters based on likelihood and clustering performance.
Compared custom PyTorch implementation with scikit-learn’s optimized GMM results.
