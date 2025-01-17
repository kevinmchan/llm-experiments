from torchtune.datasets import chat_dataset
from torchtune.models.qwen2_5 import qwen2_5_tokenizer


tokenizer = qwen2_5_tokenizer(
    path="../model/Qwen2.5-0.5B-Instruct/vocab.json",
    merges_file="../model/Qwen2.5-0.5B-Instruct/merges.txt",
)

ds = chat_dataset(
    tokenizer=tokenizer,
    source="json",
    split="train",
    conversation_column="messages",
    conversation_style="openai",
    # train_on_input=True,
    train_on_input=False,
    data_files="../data/test.json",
    # new_system_prompt="You're a helpful assistant."
)

tokenized_dict = ds[0]
tokens, labels = tokenized_dict["tokens"], tokenized_dict["labels"]

print([
    (
        token,
        tokenizer.decode([token]),
        label
    )
    for (token, label) in zip(tokens, labels)
])