from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer(custom_attr_name_dict={}, nproc=16)


print("Tokenizer start!!")
tk.tokenize_data(data_directory="/path/to/loom/files/", 
                 output_directory="/path/to/output/directory", 
                 output_prefix="output_prefix"
)
print("Tokenizer finished!!")