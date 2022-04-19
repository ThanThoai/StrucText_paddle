import torch
from transformers import AutoModel, AutoTokenizer, RobertaModel

phobert = AutoModel.from_pretrained("phobert-base")
torch_tokenizer = AutoTokenizer.from_pretrained("phobert-base")
torch_tokenizer.save_pretrained("phobert-base")
phobert.eval()

text = "Tôi là sinh_viên trường đại_học Công_nghệ ."

torch_inputs = torch_tokenizer(text, return_tensors="pt")
torch_outputs = phobert(**torch_inputs)

torch_logits = torch_outputs[0]
torch_array = torch_logits.cpu().detach().numpy()
print("torch_prediction_logits shape:{}".format(torch_array.shape))
print("torch_prediction_logits:{}".format(torch_array))


paddle_model_name = "pd_phobert-base"

import paddle
import paddlenlp
from paddlenlp.transformers.roberta.modeling import *
from paddlenlp.transformers.roberta.tokenizer import *
import numpy as np

# paddle_model = BertForPretraining.from_pretrained(paddle_model_name)
paddle_model = RobertaModel.from_pretrained(paddle_model_name)
paddle_tokenizer = RobertaTokenizer.from_pretrained(paddle_model_name)
paddle_model.eval()

paddle_inputs = paddle_tokenizer(text)
paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}
paddle_outputs = paddle_model(**paddle_inputs)

paddle_logits = paddle_outputs[0]
paddle_array = paddle_logits.numpy()
print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
print("paddle_prediction_logits:{}".format(paddle_array))


# the output logits should have the same shape
assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
diff = torch_array - paddle_array
print(np.amax(abs(diff)))