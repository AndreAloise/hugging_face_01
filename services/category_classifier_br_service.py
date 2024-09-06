import pandas as pd
from pandas import DataFrame

pd.set_option('display.max_colwidth', None)


class CategoryClassifierBrService:
	def __init__(self):
		self.dados = pd.read_csv('data_files/inputs/descricoes_produtos.csv')

	@staticmethod
	def remove_description_from_classification(classification):
		result = pd.DataFrame(classification).drop(['sequence'], axis=1)
		return result

	def get_dados(self):
		return self.dados

	@staticmethod
	def save_data_to_txt(data: DataFrame, file_name='data_files/outputs/descricoes_produtos_output.txt'):
		with open(file_name, 'w', encoding='utf-8') as f:
			f.write(data.to_string(index=False))

	def add_category_column_to_data(self, max_categories_func):
		"""
			Add new column 'Categoria' to the csv data, by a function
		"""
		self.dados['Categoria'] = self.dados['Descrição'].apply(max_categories_func)
