# MV-MoE

MV-MoE: A Multi-View Mixture-of-Expert Tuning Method for Medical LLMs

## Quick Start

### 1.Clone the Repository


```bash
git clone https://github.com/yanyongyu/MV-MoE.git
```

### 2.Install Dependencies


```bash
cd MV-MoE
uv sync
```

### 3. Code Explanation

- #### Dataset Download Page: https://tianchi.aliyun.com/dataset/165436

- #### Dataset Preprocessing
  ```
  python MV-MoE-master/src/dataset/cblue/__init__.py
  ```

- #### Environment Configuration:

  ```
  export ENHANCEMENT="${ENHANCEMENT:-understanding}"
  export ATT_SIZE="${ATT_SIZE:-2}"                
  export FFN_SIZE="${FFN_SIZE:-1024}"             
  export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"      
  ```
- #### Comparison Methods & Training Commands

	##### LoRA

    - Qwen

      ```bash
      python -m src.train.qwen.lora 2>&1 | tee train.log
      ```
    - Internlm
  
      ```bash
      python -m src.train.internlm.lora 2>&1 | tee train.log
      ```
    ##### xLoRA
  
    - Qwen
  
      ```bash
      python -m src.train.qwen.xlora 2>&1 | tee train.log
      ```
    - Internlm
  
      ```bash
      python -m src.train.internlm.xlora 2>&1 | tee train.log
      ```
    ##### Finetuning
  
    - Qwen
    
      ```bash
      python -m src.train.qwen.finetune 2>&1 | tee train.log
      ```
    - Internlm
  
      ```bash
      python -m src.train.internlm.finetune 2>&1 | tee train.log
      ```
    ##### AF Adapter
  
    - Qwen
  
      ```bash
      python -m src.train.qwen.enhance 2>&1 | tee train.log
      ```
    - Internlm
  
      ```bash
      python -m src.train.internlm.enhance 2>&1 | tee train.log
      ```
    ##### AF Experts
  
    - Qwen
  
      ```bash
      python -m src.train.qwen.adapter 2>&1 | tee train.log
      ```
    - Internlm
  
      ```python
      python -m src.train.internlm.adapter 2>&1 | tee train.log
      ```
    ##### SFT Experts
  
    - Qwen
  
      ```python
      python -m src.train.qwen.merged 2>&1 | tee train.log
      ```
    - Internlm
  
      ```python
      python -m src.train.internlm.merged 2>&1 | tee train.log
      ```
    ##### MAF Adapter（ours）
  
    - Qwen
  
      ```python
      python -m src.train.qwen.gate 2>&1 | tee train.log
      ```
    - Internlm
  
      ```python
      python -m src.train.internlm.gate 2>&1 | tee train.log
      ```
  
    #####  Adapters Merging Script
    
    ```python
    python -m src.train.merge base_model_path adapter1_path adapter2_path ... output_dir
    ```
