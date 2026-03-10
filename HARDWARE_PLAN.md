# HalluGuard Hardware Reality Plan

## 1. Current hardware environment
Development machine: Mac  
Primary execution target: local development on Mac  
Preferred device: MPS if available, otherwise CPU

## 2. Realistic project operations
The following operations are considered realistic on the current hardware:
1. Running a small LLaMA model locally for inference
2. Extracting hidden states during generation
3. Computing token-level, span-level, and overall hallucination scores
4. Saving outputs and metadata in SQLite
5. Building a small labeled training dataset
6. Training a lightweight probe model such as Logistic Regression or a small MLP
7. Running evaluation on a moderate-sized held-out test set

## 3. Operations out of scope due to hardware limits
The following operations are not realistic for this phase:
1. Full fine-tuning of the LLaMA base model
2. Large-scale LoRA or adapter fine-tuning
3. Training large deep detection models on top of full hidden-state sequences
4. Comparing many large LLM families on the same machine
5. Large benchmark processing that exceeds local memory and time constraints
6. Production-grade cloud-scale deployment from the local machine

## 4. Official training strategy
The base LLaMA model will be used only for:
- response generation
- hidden-state extraction
- feature generation

A separate lightweight hallucination detector will be trained using:
- token hidden-state features
- optional SEP and HalluShift scores

## 5. Approved probe models
The following probe models are allowed in this phase:
1. Logistic Regression
2. Small MLP
3. Other lightweight classical or shallow neural models if needed

## 6. Rejected model training strategies
The following are not part of the main implementation:
1. Full LLM fine-tuning
2. Reinforcement learning based tuning
3. Large sequence-to-sequence hallucination detector training
4. Multi-GPU or distributed training

## 7. Practical resource rule
If a step is too slow, memory-heavy, or unstable on the current machine, the project must reduce:
- dataset size
- batch size
- feature dimensionality
- experiment count

The project must not change the core research objective just to chase larger-scale training.

## 8. Final hardware-aware research position
This project is designed as a realistic hidden-state probing system on consumer-grade hardware.
The main research contribution comes from fine-grained hidden-state based hallucination detection, not from large-scale model fine-tuning.