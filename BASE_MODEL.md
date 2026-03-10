# HalluGuard Fixed Base Model

## Selected base model
Model family: LLaMA  
Exact model name: meta-llama/Llama-3.2-1B-Instruct

## Fixed experiment rule
All dataset generation, hidden-state extraction, probe training data creation, validation, testing, and demonstrations must use this same base model.

## Tokenizer
Tokenizer: paired tokenizer from the same model checkpoint

## Inference mode
Inference location: local machine  
Execution mode: Python local inference  
Device: MPS if available, otherwise CPU

## Hidden-state definition
Hidden states used in this project are the final-layer token hidden states captured during autoregressive generation.

## Scope note
The project does not compare multiple base LLMs in this phase. Model comparison is out of scope for the core research implementation.