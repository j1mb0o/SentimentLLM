# %%
import pandas as pd
import os 
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %%
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import tqdm
import evaluate

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device

# %%
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
# # Imbalanced Dataset

# %%
test_dataset = pd.read_csv(os.path.join("data-imbalance", "test-imbalance.csv"))
len(test_dataset)

# %% [markdown]
# ## Model trained on imbalanced data

# %%
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("J1mb0o/semantic-bert-imbalanced-dataset")
model = AutoModelForSequenceClassification.from_pretrained("J1mb0o/semantic-bert-imbalanced-dataset").to(device)

predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_class.append(predicted_class_id)
    true_class.append(label)
    

# %%
results_unbalanced = model_evaluation(predicted_class=predicted_class, true_class=true_class)


# %%
print(results_unbalanced)

# %% [markdown]
# ## Model trained on Balanced Dataset

# %%
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("J1mb0o/semantic-bert-balanced-dataset")
model = AutoModelForSequenceClassification.from_pretrained("J1mb0o/semantic-bert-balanced-dataset").to(device)

predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_class.append(predicted_class_id)
    true_class.append(label)
    

# %%
results_balanced = model_evaluation(predicted_class=predicted_class, true_class=true_class)


# %%
print(results_balanced)

# %% [markdown]
# ## Mistral 7b Instruct

# %%
def create_prompt1(text):
    return f"""Your task is to classify a tweet sentiment as positive, negative or neutral only. 
    
    Tweet: {text}
    
    Just generate the JSON object without explanations. Don't forget to close the JSON object with a curly bracket.
    """

def create_prompt2(text):
    return f"""Your task is to classify a tweet sentiment as positive, negative or neutral only. 
    Think step by step. 
    
    Tweet: {text}
    
    Just generate the JSON object without explanations. Don't forget to close the JSON object with a curly bracket.
    """


# %%
from langchain.llms import Ollama
import string
import json
llm = Ollama(model="mistral", temperature=0)


# %% [markdown]
# ### Evaluate on first prompt

# %%
predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    prompt = create_prompt1(text)
    response = llm(prompt)
    response = json.loads(response)
    predicted_class.append(response["sentiment"])
    true_class.append(label)


# %%
for i in predicted_class:
    if i not in ["positive", "negative", "neutral"]:
        print(i)

# %%
predicted_class_id = [["negative", "neutral", "positive"].index(i) for i in predicted_class]

# %%
mistral_results_prompt1 = model_evaluation(predicted_class=predicted_class_id, true_class=true_class)

# %%
# import string
# import json
# text = test_dataset.sample(1)
# print(text["tweet"].values[0], text["label"].values[0])
# prompt = create_prompt(text["tweet"].values[0])
# response = llm(prompt)
# # response = ''.join(ch for ch in response.strip().lower() if ch not in string.punctuation)
# response = json.loads(response)
# print(response["sentiment"])


# %%
mistral_results_prompt1

# %% [markdown]
# ### Evaluate on second prompt

# %%
predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    prompt = create_prompt2(text)
    response = llm(prompt)
    response = json.loads(response)
    predicted_class.append(response["sentiment"])
    true_class.append(label)


# %%
for i in predicted_class:
    if i not in ["positive", "negative", "neutral"]:
        print(i)

# %%
predicted_class_id = [["negative", "neutral", "positive"].index(i) for i in predicted_class]

# %%
mistral_results_prompt2 = model_evaluation(predicted_class=predicted_class_id, true_class=true_class)

# %%
mistral_results_prompt2

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
                      data=[["DistilBERT (unbalanced)", 
                              results_unbalanced[0]["accuracy"], 
                              results_unbalanced[1]["precision"][0], 
                              results_unbalanced[1]["precision"][1],
                              results_unbalanced[1]["precision"][2],
                              results_unbalanced[2]["precision"],
                              results_unbalanced[3]["recall"][0],
                              results_unbalanced[3]["recall"][1],
                              results_unbalanced[3]["recall"][2],
                              results_unbalanced[4]["recall"], 
                              results_unbalanced[5]["f1"][0],
                              results_unbalanced[5]["f1"][1],
                              results_unbalanced[5]["f1"][2],
                              results_unbalanced[6]["f1"]],
                              ["DistilBERT (balanced)",
                              results_balanced[0]["accuracy"],
                              results_balanced[1]["precision"][0],
                              results_balanced[1]["precision"][1],
                              results_balanced[1]["precision"][2],
                              results_balanced[2]["precision"],
                              results_balanced[3]["recall"][0],
                              results_balanced[3]["recall"][1],
                              results_balanced[3]["recall"][2],
                              results_balanced[4]["recall"],
                              results_balanced[5]["f1"][0],
                              results_balanced[5]["f1"][1],
                              results_balanced[5]["f1"][2],
                              results_balanced[6]["f1"]],
                              ["Mistral (prompt 1)",
                              mistral_results_prompt1[0]["accuracy"],
                              mistral_results_prompt1[1]["precision"][0],
                              mistral_results_prompt1[1]["precision"][1],
                              mistral_results_prompt1[1]["precision"][2],
                              mistral_results_prompt1[2]["precision"],
                              mistral_results_prompt1[3]["recall"][0],
                              mistral_results_prompt1[3]["recall"][1],
                              mistral_results_prompt1[3]["recall"][2],
                              mistral_results_prompt1[4]["recall"],
                              mistral_results_prompt1[5]["f1"][0],
                              mistral_results_prompt1[5]["f1"][1],
                              mistral_results_prompt1[5]["f1"][2],
                              mistral_results_prompt1[6]["f1"]],
                              ["Mistral (prompt 2)",
                              mistral_results_prompt2[0]["accuracy"],
                              mistral_results_prompt2[1]["precision"][0],
                              mistral_results_prompt2[1]["precision"][1],
                              mistral_results_prompt2[1]["precision"][2],
                              mistral_results_prompt2[2]["precision"],
                              mistral_results_prompt2[3]["recall"][0],
                              mistral_results_prompt2[3]["recall"][1],
                              mistral_results_prompt2[3]["recall"][2],
                              mistral_results_prompt2[4]["recall"],
                              mistral_results_prompt2[5]["f1"][0],
                              mistral_results_prompt2[5]["f1"][1],
                              mistral_results_prompt2[5]["f1"][2],
                              mistral_results_prompt2[6]["f1"]],
                                                  
                        ]
                      )

# %%
resutls_df.to_csv("results-unbalanced.csv", index=False)
resutls_df.to_latex("results-unbalanced.tex", index=False)

# %% [markdown]
# # Balanced Dataset

# %%
test_dataset = pd.read_csv(os.path.join("data-balance", "test-balance.csv"))
len(test_dataset)

# %% [markdown]
# ## Model trained on imbalanced data

# %%
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("J1mb0o/semantic-bert-imbalanced-dataset")
model = AutoModelForSequenceClassification.from_pretrained("J1mb0o/semantic-bert-imbalanced-dataset").to(device)

predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_class.append(predicted_class_id)
    true_class.append(label)
    

# %%
results_unbalanced = model_evaluation(predicted_class=predicted_class, true_class=true_class)


# %%
print(results_unbalanced)

# %% [markdown]
# ## Model trained on Balanced Dataset

# %%
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("J1mb0o/semantic-bert-balanced-dataset")
model = AutoModelForSequenceClassification.from_pretrained("J1mb0o/semantic-bert-balanced-dataset").to(device)

predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_class.append(predicted_class_id)
    true_class.append(label)
    

# %%
results_balanced = model_evaluation(predicted_class=predicted_class, true_class=true_class)


# %%
print(results_balanced)

# %% [markdown]
# ## Mistral 7b Instruct

# %%
def create_prompt1(text):
    return f"""Your task is to classify a tweet sentiment as positive, negative or neutral only. 
    
    Tweet: {text}
    
    Just generate the JSON object without explanations. Don't forget to close the JSON object with a curly bracket.
    """

def create_prompt2(text):
    return f"""Your task is to classify a tweet sentiment as positive, negative or neutral only. 
    Think step by step. 
    
    Tweet: {text}
    
    Just generate the JSON object without explanations. Don't forget to close the JSON object with a curly bracket.
    """


# %%
from langchain.llms import Ollama
import string
import json
llm = Ollama(model="mistral", temperature=0)


# %% [markdown]
# ### Evaluate on first prompt

# %%
predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    prompt = create_prompt1(text)
    response = llm(prompt)
    response = json.loads(response)
    predicted_class.append(response["sentiment"])
    true_class.append(label)


# %%
for i in predicted_class:
    if i not in ["positive", "negative", "neutral"]:
        print(i)

# %%
predicted_class_id = [["negative", "neutral", "positive"].index(i) for i in predicted_class]

# %%
mistral_results_prompt1 = model_evaluation(predicted_class=predicted_class_id, true_class=true_class)

# %%
# import string
# import json
# text = test_dataset.sample(1)
# print(text["tweet"].values[0], text["label"].values[0])
# prompt = create_prompt(text["tweet"].values[0])
# response = llm(prompt)
# # response = ''.join(ch for ch in response.strip().lower() if ch not in string.punctuation)
# response = json.loads(response)
# print(response["sentiment"])


# %%
mistral_results_prompt1

# %% [markdown]
# ### Evaluate on second prompt

# %%
predicted_class = []
true_class = []

for text, label in tqdm.tqdm( zip(test_dataset["tweet"], test_dataset["label"])):
    prompt = create_prompt2(text)
    response = llm(prompt)
    response = json.loads(response)
    predicted_class.append(response["sentiment"])
    true_class.append(label)


# %%
for i in predicted_class:
    if i not in ["positive", "negative", "neutral"]:
        print(i)

# %%
predicted_class_id = [["negative", "neutral", "positive"].index(i) for i in predicted_class]

# %%
mistral_results_prompt2 = model_evaluation(predicted_class=predicted_class_id, true_class=true_class)

# %%
mistral_results_prompt2

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
                      data=[["DistilBERT (unbalanced)", 
                              results_unbalanced[0]["accuracy"], 
                              results_unbalanced[1]["precision"][0], 
                              results_unbalanced[1]["precision"][1],
                              results_unbalanced[1]["precision"][2],
                              results_unbalanced[2]["precision"],
                              results_unbalanced[3]["recall"][0],
                              results_unbalanced[3]["recall"][1],
                              results_unbalanced[3]["recall"][2],
                              results_unbalanced[4]["recall"], 
                              results_unbalanced[5]["f1"][0],
                              results_unbalanced[5]["f1"][1],
                              results_unbalanced[5]["f1"][2],
                              results_unbalanced[6]["f1"]],
                              ["DistilBERT (balanced)",
                              results_balanced[0]["accuracy"],
                              results_balanced[1]["precision"][0],
                              results_balanced[1]["precision"][1],
                              results_balanced[1]["precision"][2],
                              results_balanced[2]["precision"],
                              results_balanced[3]["recall"][0],
                              results_balanced[3]["recall"][1],
                              results_balanced[3]["recall"][2],
                              results_balanced[4]["recall"],
                              results_balanced[5]["f1"][0],
                              results_balanced[5]["f1"][1],
                              results_balanced[5]["f1"][2],
                              results_balanced[6]["f1"]],
                              ["Mistral (prompt 1)",
                              mistral_results_prompt1[0]["accuracy"],
                              mistral_results_prompt1[1]["precision"][0],
                              mistral_results_prompt1[1]["precision"][1],
                              mistral_results_prompt1[1]["precision"][2],
                              mistral_results_prompt1[2]["precision"],
                              mistral_results_prompt1[3]["recall"][0],
                              mistral_results_prompt1[3]["recall"][1],
                              mistral_results_prompt1[3]["recall"][2],
                              mistral_results_prompt1[4]["recall"],
                              mistral_results_prompt1[5]["f1"][0],
                              mistral_results_prompt1[5]["f1"][1],
                              mistral_results_prompt1[5]["f1"][2],
                              mistral_results_prompt1[6]["f1"]],
                              ["Mistral (prompt 2)",
                              mistral_results_prompt2[0]["accuracy"],
                              mistral_results_prompt2[1]["precision"][0],
                              mistral_results_prompt2[1]["precision"][1],
                              mistral_results_prompt2[1]["precision"][2],
                              mistral_results_prompt2[2]["precision"],
                              mistral_results_prompt2[3]["recall"][0],
                              mistral_results_prompt2[3]["recall"][1],
                              mistral_results_prompt2[3]["recall"][2],
                              mistral_results_prompt2[4]["recall"],
                              mistral_results_prompt2[5]["f1"][0],
                              mistral_results_prompt2[5]["f1"][1],
                              mistral_results_prompt2[5]["f1"][2],
                              mistral_results_prompt2[6]["f1"]],
                                                  
                        ]
                      )

# %%
resutls_df.to_csv("results-balanced.csv", index=False)
resutls_df.to_latex("results-balanced.tex", index=False)

# %% [markdown]
# # Random Forest 

# %%
import pickle

rforest = pickle.load(open("random_forest_model.sav", "rb"))


