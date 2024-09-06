from transformers import pipeline


class CategoryClassifier:
	def __init__(self):
		self.classifier = (
			pipeline('zero-shot-classification', model='facebook/bart-large-mnli'))

	def execute_prediction(self, text: str, categories: []):
		prediction = self.classifier(text, candidate_labels=categories)
		return prediction
