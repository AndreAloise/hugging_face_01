import pandas as pd
import plotly.express as px
from pandas import DataFrame
from pysentimiento.analyzer import TokenClassificationOutput, AnalyzerOutput

pd.set_option('display.max_colwidth', None)


class SentimentClassifierBrService:
	def __init__(self):
		self.dados = pd.read_csv('data_files/inputs/resenhas.csv')

	def get_dados(self):
		return self.dados

	def add_sentiment_predictions_to_data(self, predictions: (
			TokenClassificationOutput | list[TokenClassificationOutput] | AnalyzerOutput | list[AnalyzerOutput])):
		data = self.dados

		sentiments = []
		for result in predictions:
			sentiments.append(result.output)

		data['Sentimento'] = sentiments
		return data

	@staticmethod
	def save_data_to_txt(data: DataFrame, file_name='data_files/outputs/resenha_output.txt'):
		with open(file_name, 'w', encoding='utf-8') as f:
			f.write(data.to_string(index=False))

	@staticmethod
	def create_sentiment_graph(data: DataFrame):
		df_sentiment = data.groupby('Sentimento').size().reset_index(name='Contagem')
		graph_title = 'Contagem de Resenhas por Sentimento'
		graph = px.bar(df_sentiment, x='Sentimento', y='Contagem', title=graph_title)
		graph.show()
