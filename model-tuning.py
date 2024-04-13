# %%
from datasets import Dataset, DatasetDict
import pandas as pd
import os

# %% [markdown]
# # Finetune on imbalanced dataset

# %%
train = pd.read_csv(os.path.join("data-imbalance", "train-imbalance.csv"))   
val = pd.read_csv(os.path.join("data-imbalance", "dev-imbalance.csv"))

# %%
train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)

# %%
dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

# %%
dataset_dict

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# %%
def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True)

# %%
tokenized_twitter = dataset_dict.map(preprocess_function, batched=True)

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
import evaluate

accuracy = evaluate.load("accuracy")

# %%
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# %%
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id)

# %%

training_args = TrainingArguments(
    output_dir="semantic-bert-imbalanced-dataset",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_twitter["train"],
    eval_dataset=tokenized_twitter["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()

# %% [markdown]
# # Finetuning on balanced dataset

# %%
train = pd.read_csv(os.path.join("data-balance", "train-balance.csv"))   
val = pd.read_csv(os.path.join("data-balance", "dev-balance.csv"))

# %%
train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)

# %%
dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

# %%
dataset_dict

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# %%
def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True)

# %%
tokenized_twitter = dataset_dict.map(preprocess_function, batched=True)

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
import evaluate

accuracy = evaluate.load("accuracy")

# %%
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# %%
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id)

# %%

training_args = TrainingArguments(
    output_dir="semantic-bert-balanced-dataset",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_twitter["train"],
    eval_dataset=tokenized_twitter["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()

# %% [markdown]
# # Random Forest

# %%
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# %%
train_dataset = pd.read_csv(os.path.join("data-imbalance", "train-imbalance.csv"))

# %%
X_train, y_train = train_dataset["tweet"], train_dataset["label"]

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42)),
])

# %%
text_clf.fit(X_train, y_train)

# %% [markdown]
# # Eval random forest

# %%
import evaluate
def model_evaluation(predicted_class=None, true_class=None):
    if predicted_class is None or true_class is None:
        raise ValueError("predicted and true must be not None")
    
    accuracy = evaluate.load("accuracy")
    acc = accuracy.compute(predictions=predicted_class, references=true_class, )
    
    precision = evaluate.load("precision")
    prec = precision.compute(predictions=predicted_class, references=true_class, average=None)
    prec_weigted = precision.compute(predictions=predicted_class, references=true_class, average="macro")

    recall = evaluate.load("recall")
    rec = recall.compute(predictions=predicted_class, references=true_class, average=None)
    rec_weigted = recall.compute(predictions=predicted_class, references=true_class, average="macro")

    f1 = evaluate.load("f1")
    calc_f1 = f1.compute(predictions=predicted_class, references=true_class, average=None)
    f1_weighted = f1.compute(predictions=predicted_class, references=true_class, average="macro")
    return acc, prec, prec_weigted, rec, rec_weigted, calc_f1, f1_weighted
    # return acc

# %% [markdown]
# ## Unbalanced test set

# %%
test_set_unb = pd.read_csv(os.path.join("data-imbalance", "test-imbalance.csv"))
X_test_unb, y_test_unb = test_set_unb["tweet"], test_set_unb["label"]

# %%
predicted_unb = text_clf.predict(X_test_unb)

# %%
unbalanced_eval = model_evaluation(predicted_unb, y_test_unb)

# %%
unbalanced_eval

# %%
import pandas as pd

# %% [markdown]
# ## Balanced

# %%
test_set_bal = pd.read_csv(os.path.join("data-balance", "test-balance.csv"))
X_test_bal, y_test_bal = test_set_bal["tweet"], test_set_bal["label"]

# %%
balanced_predicted = text_clf.predict(X_test_bal)

# %%
balanced_eval = model_evaluation(balanced_predicted, y_test_bal)

# %% [markdown]
# ## Export results to a dataframe

# %%
resutls_df = pd.DataFrame(columns=["Model", 
                      "Accuracy", 
                      "Pre_{Negative}", 
                      "Pre_{Neu}", 
                      "Pre_{Pos}", 
                      "Pre_{Weighted}", 
                      "Rec_{Negative}", 
                      "Rec_{Neu}", 
                      "Rec_{Pos}", 
                      "Rec_{Weighted}", 
                      "F1_{Negative}", 
                      "F1_{Neu}", 
                      "F1_{Pos}", 
                      "F1_{Weighted}"],
                      data=[["Random Forest (unbalanced)", 
                              unbalanced_eval[0]["accuracy"], 
                              unbalanced_eval[1]["precision"][0], 
                              unbalanced_eval[1]["precision"][1],
                              unbalanced_eval[1]["precision"][2],
                              unbalanced_eval[2]["precision"],
                              unbalanced_eval[3]["recall"][0],
                              unbalanced_eval[3]["recall"][1],
                              unbalanced_eval[3]["recall"][2],
                              unbalanced_eval[4]["recall"], 
                              unbalanced_eval[5]["f1"][0],
                              unbalanced_eval[5]["f1"][1],
                              unbalanced_eval[5]["f1"][2],
                              unbalanced_eval[6]["f1"]],
                              ["Random Forest (balanced)",
                              balanced_eval[0]["accuracy"],
                              balanced_eval[1]["precision"][0],
                              balanced_eval[1]["precision"][1],
                              balanced_eval[1]["precision"][2],
                              balanced_eval[2]["precision"],
                              balanced_eval[3]["recall"][0],
                              balanced_eval[3]["recall"][1],
                              balanced_eval[3]["recall"][2],
                              balanced_eval[4]["recall"],
                              balanced_eval[5]["f1"][0],
                              balanced_eval[5]["f1"][1],
                              balanced_eval[5]["f1"][2],
                              balanced_eval[6]["f1"]],
                                                  
                        ]
                      )

# %%
resutls_df.round(4)

# %%
resutls_df.to_latex("random_forest.tex", index=False, float_format="%.4f")


