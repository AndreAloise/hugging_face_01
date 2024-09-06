from transformers import pipeline


class SentimentClassifier:
	def __init__(self):
		self.feeling_classifier = (
			pipeline(
				'sentiment-analysis',
				model='distilbert-base-uncased-finetuned-sst-2-english'
			)
		)

	def execute_feeling(self, feeling: str):
		return self.feeling_classifier(feeling)
