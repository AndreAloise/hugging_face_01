from liqfit.models import T5ForZeroShotClassification
from liqfit.pipeline import ZeroShotClassificationPipeline
from transformers import T5Tokenizer


class CategoryClassifierBr:
	def __init__(self):
		self.model = T5ForZeroShotClassification.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')
		self.tokenizer = T5Tokenizer.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')
		self.classifier = (
			ZeroShotClassificationPipeline(model=self.model, tokenizer=self.tokenizer,
			                               hypothesis_template='{}', encoder_decoder=True))

	def get_classifier_of_description(self, description: str, categories: []):
		"""
			Defines just one category classification for the product description
		"""
		classification = self.classifier(description, categories, multi_label=False)
		return classification
