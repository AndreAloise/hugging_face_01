from liqfit.models import T5ForZeroShotClassification
from liqfit.pipeline import ZeroShotClassificationPipeline
from transformers import T5Tokenizer


class CategoryClassifierBr:
	def __init__(self, categories):
		self.categories = categories
		self.model = T5ForZeroShotClassification.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')
		self.tokenizer = T5Tokenizer.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')
		self.classifier = (
			ZeroShotClassificationPipeline(model=self.model, tokenizer=self.tokenizer,
			                               hypothesis_template='{}', encoder_decoder=True))

	def get_classifier_of_description(self, description: str):
		"""
			Defines just one category classification for the product description
		"""
		classification = self.classifier(description, self.categories, multi_label=False)
		return classification

	def get_max_score_categories(self, description):
		"""
			Get the max score categories from a category classification.
		"""
		classification = self.get_classifier_of_description(description)
		max_categories = (
			max(zip(classification['labels'], classification['scores']), key=lambda x: x[1]))[0]
		return max_categories
