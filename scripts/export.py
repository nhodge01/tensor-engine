from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
onnx_path = "onnx/"

# Export and save the model
model = ORTModelForFeatureExtraction.from_pretrained(model_id)
model.save_pretrained(onnx_path)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(onnx_path)