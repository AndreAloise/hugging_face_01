import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from pandas import DataFrame, Series
from wordcloud import WordCloud


class CloudWordsService:
	def __init__(self):
		self.stopwords = []

	def set_stopwords_br(self):
		nltk.download('stopwords')
		self.stopwords = set(stopwords.words('portuguese'))

	def create_cloud_words(self, data: DataFrame, data_column: str, sentiment: str):
		textos_sentimento = self._filter_data(data, data_column, sentiment)
		texto_filtrado = self._filter_text_data_by_stopwords(textos_sentimento)

		cloud_words = WordCloud(width=800, height=500, max_words=50).generate(texto_filtrado)
		plt.figure(figsize=(10, 7))
		plt.imshow(cloud_words, interpolation='bilinear')
		plt.axis('off')
		plt.show()

	@staticmethod
	def _filter_data(data: DataFrame, data_column: str, sentiment: str) \
			-> Series | None | DataFrame:
		"""
			Filter data by single sentiment
		"""
		query_expression = f"Sentimento == '{sentiment}'"
		filtered_sentiments = data.query(query_expression)[data_column]
		return filtered_sentiments

	def _filter_text_data_by_stopwords(self, sentiment_texts) -> str:
		"""
			Filter data texts by stopwords
		"""
		joined_texts_with_white_space = " ".join(sentiment_texts)
		words = joined_texts_with_white_space.split()
		filtered_words = [word for word in words if word not in self.stopwords]
		filtered_text = " ".join(filtered_words)
		return filtered_text
