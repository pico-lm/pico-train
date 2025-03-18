# 🚀 **Pico Train**

Pico Train is a lightweight framework for training language models—from tiny-scale (~1M parameters) to mid-scale (~1B parameters)—with built-in rich checkpointing that captures activations, gradients, and model states, enabling detailed learning dynamics research.

Our **suite of pre-trained models** is already publicly available on our [Hugging Face organization](https://huggingface.co/pico-lm), and a dedicated companion library for advanced analysis—[**pico-analyze**](https://github.com/pico-lm/pico-analyze)—is fully released for deeper checkpoint studies.

> For a **detailed run-through**, check out the **full tutorial** on our website at [picolm.io](https://picolm.io).

---

## **Key Features**

1. **Pico Decoder: LLAMA-style Transformer Architecture**  
   - RMSNorm, RoPE, multi-head self-attention with KV-cache, and SwiGLU activations  
   - Currently supports the **pico-decoder** model, with future expansions planned (pico-diffusion, pico-statespace, etc.)

2. **Comprehensive Checkpoints**  
   - Saves model states, optimizer states, and training metadata  
   - Enriched with **activation and gradient** snapshots for interpretability  

3. **Focused Scale Range**  
   - Optimized to train models from **1M to 1B parameters**, where learning dynamics research is most viable  

4. **Clean, Pre-tokenized Data**
   - Uses a pre-tokenized, pre-shuffled version of [Dolma](https://allenai.org/dolma) that we make available on [Hugging Face](https://huggingface.co/datasets/pico-lm/pretokenized-dolma)  
   - Facilitates training models using identical data for **consistency** and **comparability**

6. **Research Ready**  
   - Minimal, well-documented code suitable for **forking and tailoring**  
   - Logs essential metrics (e.g. perplexity) throughout training  
   - Works seamlessly with [pico-analyze](https://github.com/pico-lm/pico-analyze) for advanced post-training interpretation

---

## **Training Philosophy**

All models in the Pico suite (both pre-trained and user-trained):

- Employ **identical architectures** and **optimizer settings**  
- **Share** the same data order and tokens  
- Automatically log **rich checkpoint data** (including activations, gradients)  
- Facilitate **direct cross-scale comparisons**

This uniformity means you can isolate model size as the primary variable, giving you clearer insights into **how model capacity affects learning**.

---

## **Resources**

- **Pre-trained Models** (1M–1B parameters), publicly hosted on [Hugging Face](https://huggingface.co/pico-lm)
- **Pre-tokenized Datasets** for straightforward streaming-based training  
- **Extensive Checkpoints** logging activation and gradient snapshots  
- **Evaluation Metrics** (perplexity and more) tracked at each checkpoint

---

## **Core Components**

- **Pico-Decoder Model**  
  - LLAMA-style auto-regressive transformer  
  - RMSNorm  
  - RoPE (Rotary Positional Embeddings)  
  - Multi-head attention with KV-cache  
  - SwiGLU activation  
  
  *Future plans include additional architectures like pico-diffusion and pico-statespace.*

- **Training & Checkpointing**  
  - Automatic storage of model and optimizer states  
  - Periodic hooks for saving **learning dynamics** (activations, gradients)  
  - Optional logging to Weights & Biases

- **Config-Driven Setup**  
  - Specify architecture, optimizer, dataset, and logging settings in YAML  
  - Straightforward to extend or modify

---

## **Quick Start**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/pico-lm/pico-train
   cd pico
   ```

2. **Configure Environment**

   Create a `.env` file at the root with your Hugging Face and Weights & Biases tokens:
   ```bash
   export HF_TOKEN=your_huggingface_token
   export WANDB_API_KEY=your_wandb_key
   ```

3. **Install Dependencies**

   ```bash
   source setup.sh
   ```
   This script checks your environment, installs necessary tools, and sets up a Poetry virtual environment.

4. **Train Your Model Suite**

   - Edit (or create) a config file (e.g., `configs/demo.yaml`) to specify your architecture and training preferences.
   - Then run:
     ```bash
     poetry run train --config_path configs/demo.yaml
     ```
   - This launches training, automatically checkpointing states and saving learning dynamics data.

5. **Explore Checkpoints**
   - By default, checkpoints are stored under `runs/YOUR_RUN_NAME/checkpoints/`.
   - Each checkpoint contains:
     - **Model state** (PyTorch + Hugging Face formats)
     - **Optimizer state**
     - **Gradients and activations** for interpretability
     - **Evaluation logs** (e.g. perplexity) and metrics

---

## **Repository Structure**

- **`src/model/pico_decoder.py`**  
  - Core LLAMA-style decoder implementation (attention, RMSNorm, RoPE, etc.)

- **`src/training/trainer.py`**  
  - Main training loop  
  - Manages distributed and multi-node settings  
  - Collects/logs metrics  
  - Orchestrates checkpoint saving

- **`src/checkpointing`**  
  - Logic for saving model states, gradients, activations  
  - Tools for uploading checkpoints to Hugging Face

- **`src/config`**  
  - Flexible Dataclass-based config system (model and training hyperparameters, checkpointing, logging)

- **`configs/demo.yaml`**  
  - Example config with default values for quick experimentation

---

## **Advanced Analysis with Pico Analyze**

For deeper checkpoint analysis—comparing gradients, tracking representation shifts, measuring sparsity—use our companion repository [**pico-analyze**](https://github.com/pico-lm/pico-analyze). It automatically processes **pico-train** checkpoints and applies advanced metrics like **CKA**, **PWCCA**, **Gini**, **Hoyer**, and more to reveal **how** your models learn over time.

---

## **License**

Pico is open-source under the [Apache License 2.0](LICENSE).

---

## **Citation**

If you use **Pico** in your research, please cite:

```bibtex
@software{pico2025,
    author = {Diehl Martinez, Richard},
    title = {Pico: A Lightweight Framework for Studying Language Model Learning Dynamics},
    year = {2025},
    url = {https://github.com/pico-lm}
}
```

**Happy Training!** For more information and tutorials, visit our website at [picolm.io](https://picolm.io).
```
