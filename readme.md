# NLPDL Final Project (2022 Autumn)
## Answer-Aware Contrastive-learning based Chinese QG (base model: mT5-base)

Refine according to the paper **Contrastive Learning with Adversarial Perturbations for Conditional Text Generation**[[Paper]](https://openreview.net/forum?id=Wga_hrCa3P3)

The paper originally generate positive and negative samples from the anchor example, which I find hard to reach a better performance. To solve this issue, first of all, I warm up the model. Then I introduce contrastive loss gradually to 1 to receive better performance. You can check the code for detailed information.

The datasets should be arranged in jsonlines format. During training, the input follows this format:
```
"问题生成：..CONTEXT REGION..[HL]..ANSWER REGION..[HL]..CONTEXT REGION.."
```
while the generated question is a natural chinese representation.
## Dependecies
* python >= 3.6
* pytorch == 1.4
* transformers == 3.0.2
* sentencepiece
* datasets
* evaluate
* rouge_score
* bert_score
* sacrebleu
* jsonlines
* tqdm
* numpy

## Data Arrangement
```
cd qg_data
```
please use your dataset to replace the three sample files. If you don't want to evaluate in the training process, please set "do_eval" to false in configs.json.

## Run Baseline
* First set "adv": false in configs.json. 
* If you want to report results to [Weights & Biases](https://wandb.ai/site), please set "wandb_name" as your account name and "report_to": "wandb".  
* Also, remember to appoint the project name & task_name.
* Finally, check the data directory. This includes evaluating & predicting processes.

```
cd ../code
python main.py configs.json
```

# Run Contrastive Learning Method
* Set "adv": true in configs.json. 
* If you want to report results to [Weights & Biases](https://wandb.ai/site), please set "wandb_name" as your account name and "report_to": "wandb". 
* Also, remember to appoint the project name & task_name.
* Finally, check the data directory. This includes evaluating & predicting processes.
```
cd ../code
python main.py configs.json
```
