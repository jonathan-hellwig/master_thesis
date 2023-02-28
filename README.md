# Relations between variants of stochastic gradient descent and stochastic differential equations

In recent years, deep learning has caught the interest of many researchers. The progress in this field can be observed in its applications in robotics, medicine, physics and even in new generative AI technologies like ChatGPT and StableDiffusion. However, while the practical relevance of deep learning is undeniable, some underlying mechanisms and properties of these deep learning models are not well understood from a theory perspective. One particular topic of interest are the dynamics of parameters during the training process of neural networks. Most modern deep neural networks are trained using a variant of the prototypical stochastic gradient descent method (SGD). A recent work by Li et al. introduces a time-continuous model for the dynamics of SGD. In my thesis, I set out to investigate the applicability of this model to deep neural network. As a starting point, I used the work by Li et al. that introduces an efficient algorithm to simulate this time-continuous model.
