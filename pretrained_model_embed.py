# # !pip install fair-esm

# import esm
# print(dir(esm))

# import torch
# import esm

# # Load the ESM model and alphabet using an alternative method
# model, alphabet = esm.pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
# batch_converter = alphabet.get_batch_converter()

# import pandas as pd
# df = pd.read_csv('Ab_CoV_MIT_dataset.csv')
# df.reset_index(inplace=True)
# print(df)

# import pandas as pd
# import re

# # Function to check for invalid characters
# def has_invalid_chars(sequence):
#     return bool(re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', sequence))

# # Apply the function to each of the specified columns and filter out rows with invalid characters
# columns_to_check = ['VH sequence', 'VL sequence', 'antigen_seq']
# for col in columns_to_check:
#     df = df[~df[col].apply(has_invalid_chars)]

# print(df)

# import re

# # Function to check for invalid characters
# def has_invalid_chars(sequence):
#     return bool(re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', sequence))

# # Check sequences for invalid characters
# for index, row in df.iterrows():
#     if has_invalid_chars(row['VH sequence']):
#         print(f"Invalid characters found in sequence at index {index}: {row['VH sequence']}")

# # Extract sequences from the DataFrame
# sequences = [(row['index'], row['VH sequence']) for _, row in df.iterrows()]
# # print(sequences)


# try:
#     batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
#     # print("Batch Labels:", batch_labels)
#     # print("Batch Strings:", batch_strs)
#     # print("Batch Tokens:", batch_tokens)
# except Exception as e:
#     print(f"Error during batch conversion: {e}")

# # Convert tensors to float32 if they are not already
# batch_tokens = batch_tokens.to(torch.int32)

# with torch.no_grad():
#     results = model(batch_tokens, repr_layers=[33], return_contacts=True)
# token_representations = results["representations"][33]


# sequence_embeddings = []
# for i, (_, seq) in enumerate(sequences):
#     sequence_representation = token_representations[i, 1:len(seq) + 1].mean(0)
#     sequence_embeddings.append(sequence_representation)

# # Convert list of tensors to a NumPy array
# sequence_embeddings_np = np.stack([embedding.numpy() for embedding in sequence_embeddings])

# # Create a DataFrame from the NumPy array
# sequence_embeddings_df = pd.DataFrame(sequence_embeddings_np)

# # Save the DataFrame to a CSV file
# sequence_embeddings_df.to_csv('sequence_embeddings.csv', index=False)

# print("Sequence embeddings have been saved to 'sequence_embeddings.csv'")



# !pip install fair-esm

#############################################################################

import esm
print(dir(esm))

import torch
import esm

# Load the ESM model and alphabet using an alternative method
model, alphabet = esm.pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
batch_converter = alphabet.get_batch_converter()

import pandas as pd
df = pd.read_csv('Ab_CoV_MIT_dataset.csv')
df = df.head(10000)
df.reset_index(inplace=True)
print(df)

import re

# Function to check for invalid characters
def has_invalid_chars(sequence):
    return bool(re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', sequence))

# Apply the function to each of the specified columns and filter out rows with invalid characters
columns_to_check = ['VH sequence', 'VL sequence', 'antigen_seq']
for col in columns_to_check:
    df = df[~df[col].apply(has_invalid_chars)]

print(df)

# Check sequences for invalid characters
for index, row in df.iterrows():
    if has_invalid_chars(row['VH sequence']):
        print(f"Invalid characters found in sequence at index {index}: {row['VH sequence']}")

# Extract sequences from the DataFrame
sequences = [(row['index'], row['VH sequence']) for _, row in df.iterrows()]
# print(sequences)

try:
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    # print("Batch Labels:", batch_labels)
    # print("Batch Strings:", batch_strs)
    # print("Batch Tokens:", batch_tokens)
except Exception as e:
    print(f"Error during batch conversion: {e}")

# Convert tensors to float32 if they are not already
batch_tokens = batch_tokens.to(torch.int32)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

sequence_embeddings = []
for i, (_, seq) in enumerate(sequences):
    sequence_representation = token_representations[i, 1:len(seq) + 1].mean(0)
    sequence_embeddings.append(sequence_representation)

    # Clear memory after processing each sequence
    del batch_tokens, results, token_representations
    gc.collect()

# Convert list of tensors to a NumPy array
sequence_embeddings_np = np.stack([embedding.numpy() for embedding in sequence_embeddings])

# Create a DataFrame from the NumPy array
sequence_embeddings_df = pd.DataFrame(sequence_embeddings_np)

# Save the DataFrame to a CSV file
sequence_embeddings_df.to_csv('sequence_embeddings.csv', index=False)

print("Sequence embeddings have been saved to 'sequence_embeddings.csv'")

#############################################################################

# import esm
# import torch
# import pandas as pd
# import re
# import gc
# import numpy as np
# import psutil
# import time

# # Function to monitor and print memory and CPU usage
# def monitor_resources():
#     print(f"CPU Usage: {psutil.cpu_percent()}%")
#     memory_info = psutil.virtual_memory()
#     print(f"Memory Usage: {memory_info.percent}% ({memory_info.used / (1024 ** 3):.2f} GB used out of {memory_info.total / (1024 ** 3):.2f} GB)")

# # Load the ESM model and alphabet using an alternative method
# model, alphabet = esm.pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
# batch_converter = alphabet.get_batch_converter()

# # Load the dataset
# df = pd.read_csv('Ab_CoV_MIT_dataset.csv')
# df.reset_index(inplace=True)

# # Function to check for invalid characters
# def has_invalid_chars(sequence):
#     return bool(re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', sequence))

# # Apply the function to each of the specified columns and filter out rows with invalid characters
# columns_to_check = ['VH sequence', 'VL sequence', 'antigen_seq']
# for col in columns_to_check:
#     df = df[~df[col].apply(has_invalid_chars)]

# # Extract sequences from the DataFrame
# sequences = [(row['index'], row['VH sequence']) for _, row in df.iterrows()]

# # Process sequences in batches
# batch_size = 10  # Adjust batch size based on your memory capacity
# sequence_embeddings = []

# for i in range(0, len(sequences), batch_size):
#     batch_sequences = sequences[i:i + batch_size]
#     batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)
#     batch_tokens = batch_tokens.to(torch.int32)

#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#     token_representations = results["representations"][33]

#     for j, (_, seq) in enumerate(batch_sequences):
#         sequence_representation = token_representations[j, 1:len(seq) + 1].mean(0)
#         sequence_embeddings.append(sequence_representation)

#     # Clear memory after processing each batch
#     del batch_tokens, results, token_representations
#     gc.collect()

#     # Monitor resources after each batch
#     monitor_resources()
#     time.sleep(1)  # Optional: Add a short delay to reduce the frequency of monitoring

# # Convert list of tensors to a NumPy array
# sequence_embeddings_np = np.stack([embedding.numpy() for embedding in sequence_embeddings])

# # Create a DataFrame from the NumPy array
# sequence_embeddings_df = pd.DataFrame(sequence_embeddings_np)

# # Save the DataFrame to a CSV file
# sequence_embeddings_df.to_csv('sequence_embeddings.csv', index=False)

# print("Sequence embeddings have been saved to 'sequence_embeddings.csv'")
