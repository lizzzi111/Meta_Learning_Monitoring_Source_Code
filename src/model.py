from __future__ import annotations  # noqa: EXE002, D100

import gc

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(model_name: str | int, freeze:bool=True, analysis_name:str=None) -> AutoModelForSeq2SeqLM:  # noqa: FBT001, FBT002, E999, RUF013
    """Load T5 model and tokenizer."""
    if model_name=="CodeT5":
        model_name="Salesforce/codet5-base-multi-sum"
    elif isinstance(model_name, int):
        if analysis_name is None:
            msg = "Requires analysis name"
            raise NameError(msg)
        model_name = f"reports/{analysis_name}/trained_model"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for i, m in enumerate(model.decoder.block):
            #Only un-freeze the last n transformer blocks in the decoder
            if i+1 > 12 - 4:
                for parameter in m.parameters():
                    parameter.requires_grad = True

    return model

def load_tok(model_name: str | int, analysis_name:str=None) -> AutoTokenizer:  # noqa: RUF013
    """Load T5 tokenizer."""
    if model_name=="CodeT5":
        model_name="Salesforce/codet5-base-multi-sum"
    elif isinstance(model_name, int):
        if analysis_name is None:
            msg = "Requires analysis name"
            raise NameError(msg)
        model_name = f"reports/{analysis_name}/trained_model_{model_name}"
    return AutoTokenizer.from_pretrained(model_name, skip_special_tokens=False)


def torch_clean():  # noqa: ANN201
    """Clean up torch cache."""
    gc.collect()
    torch.cuda.empty_cache()
