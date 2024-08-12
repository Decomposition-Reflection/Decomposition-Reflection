## Prepare

1. In Hugging Face, download `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` and place the `multi-qa-MiniLM-L6-cos-v1` model in the `sentence-transformers` folder;
2. In Hugging Face, download `google/pegasus-large` and place the `pegasus-large` model in the `google` folder;
3. In Hugging Face, download `meta-llama/Llama-2-7b-chat-hf`,`lmsys/vicuna-7b-v1.5`,`google/flan-t5-small` and place them in the `model` folder;
4. In Hugging Face, download `FacebookAI/roberta-large` and place the `roberta-large` model in the `facebook` folder. Run the following command to install the required libraries:

```bash
pip install evaluate bert-score numpy transformers torch
```

## To run: (model:LLaMA,dataset:ExpertQA,method:Decomposition-Reflection)

```bash
python3 run.py
```

## To run: Ablation(without reflection:ablation1.py,ablation2.py,ablation3.py;
## without decomposition:ablation4.py)

```bash
python3 ablation/ablation1.py
python3 ablation/ablation2.py
python3 ablation/ablation3.py
python3 ablation/ablation4.py
```
