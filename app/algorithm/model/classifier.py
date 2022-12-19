import numpy as np, pandas as pd
from scipy.special import softmax
import joblib
import pprint
import json
import sys
import os, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from transformers import pipeline
import torch

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset


torch.cuda.empty_cache()


MODEL_NAME = "text_class_base_distilbert_hf"

model_params_fname = "model_params.save"
trainer_fname = "trainer.save"
tokenizer_fname = "tokenizer.save"
transformer_fname = "transformer.save"
training_args_fname = "training_args.save"
history_fname = "history.json"


pretrained_model_path = os.path.join(os.path.dirname(__file__), "pretrained_model")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device={device}")


def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}


class Classifier:
    def __init__(self, num_labels, **kwargs):
        self.num_labels = num_labels
        # self.checkpoint = "distilbert-base-uncased"

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _tokenize_function(self, df):
        return self.tokenizer(
            df["text"], padding="max_length", max_length=64, truncation=True
        )

    def fit(self, train_data, valid_data, num_train_epochs, save_path):
        # self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

        train_dataset = Dataset.from_pandas(train_data).map(
            self._tokenize_function, batched=True
        )
        valid_dataset = Dataset.from_pandas(valid_data).map(
            self._tokenize_function, batched=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            # self.checkpoint,
            pretrained_model_path,
            num_labels=self.num_labels,
        )

        # summary(self.model); sys.exit()

        self.training_args = TrainingArguments(
            output_dir=save_path,
            logging_dir=save_path,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=num_train_epochs,
            # save_total_limit=2,
            load_best_model_at_end=True,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        _ = self.trainer.train()
        return self

    def predict(self, documents):
        dataset = Dataset.from_pandas(documents, preserve_index=False)
        tokenized_datasets = dataset.map(self._tokenize_function)
        logits, _, _ = self.trainer.predict(tokenized_datasets)
        preds = softmax(logits, axis=-1)
        return preds

    def model_summary(self):
        summary(self.model)

    def save(self, model_path):

        model_params = {"num_labels": self.num_labels}
        model_params_fpath = os.path.join(model_path, model_params_fname)
        joblib.dump(model_params, model_params_fpath)

        transformer_fpath = os.path.join(model_path, transformer_fname)
        self.model.save_pretrained(transformer_fpath)

        tokenizer_fpath = os.path.join(model_path, tokenizer_fname)
        self.tokenizer.save_pretrained(tokenizer_fpath)

        training_args_fpath = os.path.join(model_path, training_args_fname)
        joblib.dump(self.training_args, training_args_fpath)

    @classmethod
    def load(ml, model_path):
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        classifier = Classifier(**model_params)

        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(model_path, transformer_fname)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, tokenizer_fname)
        )
        training_args = joblib.load(os.path.join(model_path, training_args_fname))

        classifier.model = model
        classifier.tokenizer = tokenizer
        classifier.trainer = Trainer(model=model, args=training_args)
        return classifier


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = Classifier.load(model_path)
    return model


def get_data_based_model_params(train_data, valid_data, hyper_params):
    """
    Set any model parameters that are data dependent.
    For example, number of layers or neurons in a neural network as a function of data shape.
    """
    num_of_classes = max(max(train_data["label"]), max(valid_data["label"])) + 1
    return {"num_labels": num_of_classes}
