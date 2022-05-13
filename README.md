### [COMP6248 Reproductibility Challenge]
## Are Neural Nets Modular? Inspecting the Functional Modularity of NNs using a differentiable weight mask

Paper: https://arxiv.org/abs/2010.02066 

Original Code Base: https://github.com/RobertCsordas/modules/tree/efdb8790b074862581e035c9ab5bf889440a8023

### This project includes :
- Data generation method (algorithmic data, +/* operations)
- Network pre-training (FNN) (model data can be found in `networks/cache-networks/`)
- Mask Generation (pre-generated masks can be found in `trainedmasks/`)
- Testing for Masked Network
- Calculating degree of weight sharing between two masks
