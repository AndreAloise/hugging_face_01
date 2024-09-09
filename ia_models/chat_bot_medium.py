from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatBotMedium:
	def __init__(self):
		self._tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
		self._model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

	def get_tokenizer(self):
		return self._tokenizer

	def get_model(self):
		return self._model
