from pandas import DataFrame

from ia_models.sentiment_classifier import SentimentClassifier
from ia_models.sentiment_classifier_br import SentimentClassifierBr
from services.cloud_words_service import CloudWordsService
from services.sentiment_classifier_br_service import SentimentClassifierBrService


def execute_ia_classifier():
	classifier = SentimentClassifier()
	print(f'Sentiment Analyser: {classifier.execute_feeling('I love this product')}')

	classifier_br = SentimentClassifierBr()
	text_br_01 = 'A fritadeira é sensacional, muito além do que imaginava. É linda, super funcional e muito fácil de manusear. Fácil de limpar e potente. Super recomendo.'
	print(f'Texto 01: {classifier_br.execute_prediction(text_br_01)}')

	text_br_02 = 'Após poucos meses de uso a carcaça de aço escovado começou a oxidar, demonstrando a baixa qualidade de proteção. Fora esse detalhe, o produto cumpre o prometido.'
	print(f'Texto 02: {classifier_br.execute_prediction(text_br_02)}')

	text_br_03 = 'Em menos de 1 ano parou de funcionar, enviei para assistência técnica por estar na garantia, trocaram o motor, passou a ficar menos potente e não durou 2 utilizações. Isso se repetiu várias vezes, até que desisti de ficar levando lá e queimando de novo em seguida, vi outros clientes com o mesmo problema. Não comprem!!'
	print(f'Texto 03: {classifier_br.execute_prediction(text_br_03)}')


def _get_sentiment_prediction_from_csv():
	service = SentimentClassifierBrService()
	raw_data = service.get_dados()

	classifier_br = SentimentClassifierBr()
	results = classifier_br.execute_prediction(raw_data.get('Resenha'))

	converted_data = service.add_sentiment_predictions_to_data(results)

	return converted_data


def save_data_to_txt_file(converted_data: DataFrame):
	SentimentClassifierBrService.save_data_to_txt(converted_data)


def get_sentiment_graph(converted_data: DataFrame):
	SentimentClassifierBrService.create_sentiment_graph(converted_data)


def get_cloud_words(converted_data: DataFrame):
	cloud_words_service = CloudWordsService()
	cloud_words_service.set_stopwords_br()
	cloud_words_service.create_cloud_words(converted_data, 'Resenha', 'POS')


if __name__ == '__main__':
	sentiment_data = _get_sentiment_prediction_from_csv()
	# execute_ia_classifier()
	# save_data_to_txt_file()
	# get_sentiment_graph()
	get_cloud_words(sentiment_data)
