from pysentimiento import create_analyzer
from pysentimiento.analyzer import TokenClassificationOutput, AnalyzerOutput


class SentimentClassifierBr:
	def __init__(self):
		self.feeling_classifier = create_analyzer('sentiment', 'pt')

	def execute_prediction(self, text: str) \
			-> (
					TokenClassificationOutput
					| list[TokenClassificationOutput]
					| AnalyzerOutput
					| list[AnalyzerOutput]
			):
		return self.feeling_classifier.predict(text)
