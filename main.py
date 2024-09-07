from pandas import DataFrame

from ia_models.background_removal import BackgroundRemoval
from ia_models.category_classifier import CategoryClassifier
from ia_models.category_classifier_br import CategoryClassifierBr
from ia_models.sentiment_classifier import SentimentClassifier
from ia_models.sentiment_classifier_br import SentimentClassifierBr
from interface_views.background_removal_view import BackgroundRemovalView
from services.category_classifier_br_service import CategoryClassifierBrService
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


def _get_category_prediction():
	text = 'Latest model of smartphone with 5G connectivity and 128GB internal storage'
	categories = ['electronics', 'food', 'toys', 'books']
	category_classifier = CategoryClassifier()
	predictions = category_classifier.execute_prediction(text, categories)

	print(predictions)


def get_category_classifier_br():
	# description = 'Este water gel leve e refrescante, proporciona hidratação imediata que ajuda a aliviar o repuxamento e aspereza da pele sensível.'
	description = '''A fritadeira eletrica sem óleo start fry da elgin possui um design único, capacidade para até 3,5 litros,
            potência de 1400w e revestimento antiaderente. Seu sistema de circulação de ar ultra rápido frita e economiza energia.
            Sua grelha de fritura é removível e super fácil de limpar. Ela conta com uma proteção contra super aquecimento.
            Possui controle de temperatura de 80°c a 200°c que permite você programar a temperatura de preparo para cada tipo de alimento,
            timer para até 60 minutos com aviso sonoro e desligamento automático, assim você pode deixar preparando sua refeição
            enquanto realiza outras tarefas.'''
	categories = ['beleza', 'livros', 'cozinha']
	category_classifier_br = CategoryClassifierBr(categories)
	classification = category_classifier_br.get_classifier_of_description(description)

	classification_without_sequence = CategoryClassifierBrService.remove_description_from_classification(classification)
	print(classification_without_sequence)


def _get_product_description_from_csv():
	service = CategoryClassifierBrService()
	raw_data = service.get_dados()
	CategoryClassifierBrService.save_data_to_txt(raw_data)


def add_category_to_data():
	categories = ['eletrodomésticos', 'eletrônicos', 'beleza', 'brinquedos']
	category_classifier_br = CategoryClassifierBr(categories)

	service = CategoryClassifierBrService()
	service.add_category_column_to_data(category_classifier_br.get_max_score_categories)
	updated_data = service.get_dados()
	CategoryClassifierBrService.save_data_to_txt(updated_data)


def get_cleaned_image():
	model = BackgroundRemoval()
	model.show_image('data_files/inputs/images/camera_fotografica.jpg')


def create_image_view():
	model = BackgroundRemoval()
	view = BackgroundRemovalView(model)
	view.create_gradio_interface()


if __name__ == '__main__':
	# sentiment_data = _get_sentiment_prediction_from_csv()
	# execute_ia_classifier()
	# save_data_to_txt_file()
	# get_sentiment_graph()
	# get_cloud_words(sentiment_data)
	# _get_category_prediction()
	# get_category_classifier_br()
	# _get_product_description_from_csv()
	# add_category_to_data()
	# get_cleaned_image()
	create_image_view()
