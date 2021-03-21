# cs224w-project

## Description
We propose GraphCORAL as a loss formulation for model performance improvement. 
We examine GraphCORAL on the OGB ArXiv dataset on three models: GCN, GraphSage, and IncepGCN. 
We also explore other improvements, such as FLAG. The following code trains and tests with outputs for our best performing configurations for each of our models.

## Requirements & Installation:
* PyTorch 1.8
* Torch Geometric 1.6

Install packages:
```
pip install -r requirements.txt
```

## Run best model configurations:
Run IncepGCN configuration **IncepGCN + GraphCORAL + FLAG**:
```
python train.py --dataset ogbn-arxiv --model_type incepgcn --dropedge 0 \
--epochs 500 --runs 5 --hidden_dim 128 --num_flag_steps 3 --coral_lambda 0.1 
```

Run GraphSage configuration **GraphSage + FLAG**:
```
python train.py --dataset ogbn-arxiv --model_type graphsage --dropedge 0 \
--epochs 500 --runs 5 --hidden_dim 128 --num_flag_steps 3 
```

Run GCN configuration **GCN + GraphCORAL + FLAG**:
```
python train.py --dataset ogbn-arxiv --model_type incepgcn --dropedge 0 \
--epochs 500 --runs 5 --hidden_dim 128 --num_flag_steps 3 --coral_lambda 0.01 
```


### Detailed hyperparameters:
```
IncepGCN + GraphCORAL + FLAG:             GraphSage + FLAG:              GCN + GraphCORAL + FLAG:
--model_type        incepgcn              --model_type     graphsage     --model_type       gcn                   
--dataset_name      ogbn-arxiv            --dataset_name   ogbn-arxiv    --dataset_name     ogbn-arxiv               
--runs              5                     --runs           5             --runs             5
--epochs            500                   --epochs         500           --epochs           500 
--lr                0.01                  --lr             0.01          --lr               0.01 
--num_layers        3                     --num_layers     3             --num_layers       3
--hidden_dim        128                   --hidden_dim     128           --hidden_dim       128
--dropout           0.5                   --dropout        0.5           --dropout          0.5
--dropedge          0                     --dropedge       0             --dropedge         0 
--num_flag_steps    3                     --num_flag_steps 3             --num_flag_steps   3 
--flag_step_size    0.001                 --flag_step_size 0.001         --flag_step_size   0.001 
--coral_lambda      0.1                                                  --coral_lambda     0.01 
--num_loss_layers   1                                                    --num_loss_layers  1 
--num_branches      3           
--self_loops        True          
```

 
### Reference performance for OGB-Arxiv:
| Model + Configuration      | Validation Accuracy | Test Accuracy |
| ----------- | ----------- | ----------- | 
| GCN + GraphCORAL + FLAG   |  72.92 ± 0.09       | 72.01 ± 0.16 |
| GraphSage + FLAG   | 72.91 ± 0.07     | 71.73 ± 0.16 |
| IncepGCN + GraphCORAL + FLAG   | 73.78 ± 0.09   | 72.52 ± 0.15  |
 



