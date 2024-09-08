from pandas import DataFrame

from ia_models.background_removal import BackgroundRemoval
from ia_models.category_classifier import CategoryClassifier
from ia_models.category_classifier_br import CategoryClassifierBr
from ia_models.faq_question_answer import FaqQuestionAnswer
from ia_models.sentiment_classifier import SentimentClassifier
from ia_models.sentiment_classifier_br import SentimentClassifierBr
from interface_views.background_removal_view import BackgroundRemovalView
from interface_views.faq_question_answer_view import FaqQuestionAnswerView
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


def create_faq_view():
	contexts = {
		"How do I create an account?": "You can create an account by clicking on the 'Sign Up' button on our homepage. You will need to provide your email address, create a password, and fill in some basic information about yourself. An account will help you track your orders, manage your personal settings, and speed up future transactions.",
		"Which payment methods do you accept?": "We accept a wide range of payment methods including Visa, MasterCard, American Express, Discover, PayPal, and Apple Pay. You can also pay using store credit or gift cards issued by our company.",
		"How can I track my order?": "Once your order has shipped, you will receive an email with a tracking number. You can use this number on our website's tracking page to see the current status of your delivery.",
		"Do you offer international shipping?": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary depending on the destination. All applicable customs fees, taxes, and duties are the responsibility of the customer and are calculated at checkout.",
		"How long does delivery take?": "For standard shipping, deliveries typically take between 3 to 5 business days. For expedited shipping, expect your order to arrive within 1 to 2 business days. Delivery times may vary based on your location and the time of the year.",
		"What is your return policy?": "Our return policy allows you to return products within 30 days of receiving them. Items must be in their original condition and packaging. Some items, such as perishable goods, are not eligible for return.",
		"Can I change or cancel my order after it's been placed?": "You can change or cancel your order within 24 hours of placing it without any additional charge.To make changes or cancel your order, please contact our customer service immediately.",
		"What should I do if I receive a damaged item?": "If you receive a damaged item, please contact our customer service within 48 hours of delivery to report the damage. You will need to provide your order number, the description of the damage, and photographic evidence. We will arrange for a replacement or refund as appropriate.",
		"How do I reset my password?": "If you've forgotten your password, go to the login page and click on 'Forgot Password'. Enter your email address and we will send you a link to reset your password. For security purposes, this link will expire within 24 hours."
	}
	model = FaqQuestionAnswer(contexts)
	view = FaqQuestionAnswerView(model)
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
	# create_image_view()
	create_faq_view()
