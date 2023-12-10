from einops import rearrange
import torch
from transformers import T5Tokenizer, T5EncoderModel
# Check https://huggingface.co/models
t5_setup = "t5-small"
MAX_LENGTH = 256
encoded_dims = 512

def t5_encode_text(text, max_length=MAX_LENGTH):
    """
    Encodes a sequence of text using T5 text encoder
    :return: Returns encoding and attention mask
    """
    tokenizer = T5Tokenizer.from_pretrained(t5_setup)
    model = T5EncoderModel.from_pretrained(t5_setup)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    elif torch.backends.mps.is_available(): # works for Apple Silicon
        device = torch.device("mps")
        model.to(device)
    else:
        model.to("cpu")
    
    tokenized_text = tokenizer.batch_encode_plus(text, padding="longest",
                                                 max_length=max_length, truncation=True,
                                                 return_tensors="pt")
    input_ids = tokenized_text.input_ids.to(device)
    attention_mask = tokenized_text.attention_mask.to(device)

    model.eval()

    with torch.inference_mode():
        t5_output = model(input_ids=input_ids, attention_mask=attention_mask)
        final_encoding = t5_output.last_hidden_state.detach()
    
    # wherever the encoding is masked, we make them equal to zero
    final_encoding = final_encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)

    return final_encoding, attention_mask.bool()

def get_encoded_dim():
    return encoded_dims