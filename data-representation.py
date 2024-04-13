# %%
import pandas as pd
import os
import numpy as np

# %%
train_dataset = pd.read_csv(os.path.join("downloaded", "twitter-2016train-A.tsv"), 
                            sep="\t", 
                            header=None,
                            names=["id", "label", "tweet"])
dev_dataset = pd.read_csv(os.path.join("downloaded", "twitter-2016dev-A.tsv"), 
                          sep="\t", 
                          header=None,
                          names=["id", "label", "tweet"])
test_dataset = pd.read_csv(os.path.join("downloaded", "twitter-2016devtest-A.tsv"), 
                           sep="\t", 
                           header=None,
                           names=["id", "label", "tweet"]
                           )
                           
train_dataset = train_dataset[["tweet", "label"]]
dev_dataset = dev_dataset[["tweet", "label"]]
test_dataset = test_dataset[["tweet", "label"]]

train_dataset["label"] = train_dataset["label"].apply(lambda x: 0 if x == "negative" else 1 if x == "neutral" else 2)
dev_dataset["label"] = dev_dataset["label"].apply(lambda x: 0 if x == "negative" else 1 if x == "neutral" else 2)
test_dataset["label"] = test_dataset["label"].apply(lambda x: 0 if x == "negative" else 1 if x == "neutral" else 2)

# %% [markdown]
# # Dataset statistics

# %%
import matplotlib.pyplot as plt
colors = ['red', 'orange', 'limegreen']

# %% [markdown]
# ## Imbalanced Data

# %%
train_dat_class = [sum(train_dataset["label"] == i) for i in range(3)]
plt.bar(["negative", "neutral", "positive"], train_dat_class, color=colors)
plt.title("Train Data Class Distribution")
plt.savefig(os.path.join("data-imbalance","train-class.pdf"))

# %%
test_dat_class = [sum(test_dataset["label"] == i) for i in range(3)]
plt.bar(["negative", "neutral", "positive"], test_dat_class, color=colors)
plt.title("Test Data Class Distribution")
plt.savefig(os.path.join("data-imbalance","test-class.pdf"))

# %%

dev_dat_class = [sum(dev_dataset["label"] == i) for i in range(3)]
plt.bar(["negative", "neutral", "positive"], dev_dat_class, color=colors)
plt.title("Validation Data Class Distribution")
plt.savefig(os.path.join("data-imbalance","val-class.pdf"))

# %% [markdown]
# ## Balance Data

# %%
np.random.seed(42)

# %%
balanced_test = test_dataset.groupby('label')
balanced_test = balanced_test.apply(lambda x: x.sample(balanced_test.size().min()).reset_index(drop=True))
balanced_test["label"].value_counts()
# balanced_test

balanced_train = train_dataset.groupby('label')
balanced_train = balanced_train.apply(lambda x: x.sample(balanced_train.size().min()).reset_index(drop=True))
balanced_train["label"].value_counts()

balanced_dev = dev_dataset.groupby('label')
balanced_dev = balanced_dev.apply(lambda x: x.sample(balanced_dev.size().min()).reset_index(drop=True))


# %%
train_dat_class = [sum(balanced_train["label"] == i) for i in range(3)]
plt.bar(["negative", "neutral", "positive"], train_dat_class, color=colors)
plt.title("Train Data Class Distribution")
plt.savefig(os.path.join("data-balance","train-bal-class.pdf"))

# %%
valid = [sum(balanced_dev["label"] == i) for i in range(3)]
plt.bar(["negative", "neutral", "positive"], valid, color=colors)
plt.title("Validation Data Class Distribution")
plt.savefig(os.path.join("data-balance","val-bal-class.pdf"))

# %%
test = [sum(balanced_test["label"] == i) for i in range(3)]
plt.bar(["negative", "neutral", "positive"], test, color=colors)
plt.title("Test Data Class Distribution")
plt.savefig(os.path.join("data-balance","test-bal-class.pdf"))

# %% [markdown]
# # Export the dataframes as CSV files

# %%
train_dataset.to_csv(os.path.join("data-imbalance", "train-imbalance.csv"), index=False)
dev_dataset.to_csv(os.path.join("data-imbalance", "dev-imbalance.csv"), index=False)
test_dataset.to_csv(os.path.join("data-imbalance", "test-imbalance.csv"), index=False)

# %%
balanced_test.to_csv(os.path.join("data-balance", "test-balance.csv"), index=False)
balanced_train.to_csv(os.path.join("data-balance", "train-balance.csv"), index=False)
balanced_dev.to_csv(os.path.join("data-balance", "dev-balance.csv"), index=False)


