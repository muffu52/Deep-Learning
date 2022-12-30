from config import idx_label, label_idx
from predict import predict
from preprocess_data import process_data
from train import train
import torch


def start(task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    process_data(label_idx[task], f'label_{task}', "data/train_all_tasks.csv")
    # Change the model to bert-large-uncased or any other sequence classifier
    # distilbert-base-uncased
    best_model_checkpoint = train(
        "distilbert-base-uncased", f'label_{task}', idx_label[task], label_idx[task])
    # best_model_checkpoint = "C:/Users/mufdu/Desktop/New folder/DL/New folder/SemEval-2023-Task-10-EDOS/results/checkpoint-10500"
    predict(best_model_checkpoint,
            f'data/dev_task_{task}.csv', 'task_{task}')


# start("sexist")
# start("category")
start("vector")
