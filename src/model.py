import gc

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model_tok(model_name: str, freeze:bool=True) -> list:  # noqa: FBT001, FBT002

    if model_name=="CodeT5":
        model_name="Salesforce/codet5-base-multi-sum"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=False)

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for i, m in enumerate(model.decoder.block):
            #Only un-freeze the last n transformer blocks in the decoder
            if i+1 > 12 - 4:
                for parameter in m.parameters():
                    parameter.requires_grad = True

    return model, tokenizer


def torch_clean():  # noqa: ANN201
    gc.collect()
    torch.cuda.empty_cache()
