from pysentimiento import create_analyzer


class SentimentClassifierBr:
	def __init__(self):
		self.feeling_classifier = create_analyzer('sentiment', 'pt')

	def execute_prediction(self, text: str):
		return self.feeling_classifier.predict(text)
