# SparseBERT: Efficient Transformer-based Language Model with Integrated Pruning
## High Performance Machine Learning Project

SparseBERT is a research project that incorporates pruning techniques into transformer-based language models, aimed at creating more efficient models for resource-constrained environments. This implementation provides various sparsification methods that significantly reduce model size and computational requirements while maintaining performance on downstream tasks.

## Description
Transformer architectures like BERT have set new benchmarks in natural language processing, but their extensive parameter counts and computational requirements limit their applicability in resource-constrained environments. SparseBERT addresses these challenges by implementing multiple model compression techniques that are integrated into the training process:
Our implementation uses ModernBERT-base as the foundation model, which we then enhance with the following pruning mechanisms:

- Unstructured magnitude-based pruning
- Structured head pruning
- Structured feed-forward network pruning
- NVIDIA's 2:4 structured sparsity pattern
- Progressive pruning with custom schedules
- Post-training quantization
- Loss Function Regularization (L1 and L0 norm-based)

These techniques work together to significantly reduce model size and improve inference speed while maintaining performance on downstream natural language processing tasks.


## Framework
### Hardware Framework:
Compute: NVIDIA GPUs (Tesla V100, A100) for parallel processing
Environment: NYU HPC for cloud and edge-based execution

### Software Framework:
Framework: PyTorch with the Hugging Face Transformers library
Existing Code Reuse:
- Transformer implementations from Hugging Face
- PyTorch's pruning utilities
- Custom training loops with pruning logic

### Project Milestones and Status: 
- Core SparseBERT implementation with model adaptation - Completed
- Magnitude-based pruning implementation- Completed
- Head and feed-forward network - Completed
- NVIDIA's 2:4 sparsity pattern for GPU acceleration - Completed
- Gradual pruning with custom schedules- Completed
- Post-training quantization for further size reduction- Completed
- Pruning with fine-tuning to recover performance- Completed
- Evaluation on standard GLUE task- Completed
- Comprehensive documentation and examples - Completed

### Description of Repository and Code structure and Running code:
- HPMLPruningV4.ipynb is final jupyter notebook
- Download the .ipynb above and upload it to your google drive
- Run all the cells.

## Evaluation
URL: https://gluebenchmark.com/tasks
We evaluate SparseBERT using the GLUE benchmark, focusing on tasks such as:
- Multi-Genre Natural Language Inference (MNLI): Determining whether a premise entails, contradicts, or is neutral to a hypothesis
- Quora Question Pairs (QQP): Identifying duplicate questions
- Stanford Sentiment Treebank (SST-2): Binary sentiment classification

### Our evaluation metrics include:
- Task performance (accuracy, F1 score)
- Model size reduction
- Inference speed improvement
- Memory footprint reduction

## Summary Comparison of Different Pruning Methods

| Method | Sparsity (%) | Model Size (MB) | Inference Time (ms) | Speedup | Avg GLUE Acc |
|--------|--------------|-----------------|--------------------:|--------:|-------------:|
| ✅ Baseline | 0.0 | 570.7 | 759.38 | 1.0 | **0.897** |
| Magnitude | 22.24 | 443.8 | 746.4 | 1.02 | 0.88 |
| Head | 10.38 | 511.45 | 722.39 | 1.05 | 0.882 |
| FFN | 7.81 | 526.15 | 828.82 | 0.92 | 0.87 |
| 2:4 | 37.06 | 359.2 | 664.07 | 1.14 | 0.865 |
| ✅ Progressive | 37.14 | 358.74 | 657.26 | 1.16 | **0.89** |

### Conclusions and Observations
- Progressive pruning and 2:4 structured sparsity provide the best balance of model compression and inference speed.
- Head pruning is effective for moderate pruning with minimal accuracy loss.
- pruning is simple and effective but less hardware-efficient.
- Fine-tuning after pruning can recover most of the performance loss.
- Combining pruning with quantization provides additional size reduction with minimal impact.

## Future Work
- Implementation of other pruning methods (e.g., movement pruning, lottery ticket pruning)
- Support for more base models (e.g., RoBERTa, GPT variants)
- Hardware-specific optimizations for pruned models
- Distillation-aware pruning
- Integration with TensorRT and ONNX Runtime for production deployment

## Acknowledgments
- This work was supported by computational resources from NYU HPC
- The ModernBERT-base model is from AnswerDotAI
