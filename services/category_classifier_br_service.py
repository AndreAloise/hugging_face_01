import pandas as pd


class CategoryClassifierBrService:
	def __init__(self):
		pass

	@staticmethod
	def remove_description_from_classification(classification):
		result = pd.DataFrame(classification).drop(['sequence'], axis=1)
		return result
