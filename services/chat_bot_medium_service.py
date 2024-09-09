import pandas as pd
import torch

from ia_models.chat_bot_medium import ChatBotMedium


class ChatBotMediumService:
	_key_words_status = [
		'order',
		'order status',
		'status of my order',
		'check my order',
		'track my order',
		'order update'
	]

	_key_words_finish_chat = [
		'exit,'
		'quit',
		'stop'
	]

	def __init__(self, dados_pedidos):
		self.df_status_pedidos = pd.DataFrame(dados_pedidos)

	def _check_status_pedido(self, pedido_number):
		try:
			status = self.df_status_pedidos[self.df_status_pedidos['numero_pedido'] == pedido_number]['status'].iloc[0]
			return f'The status of your order {pedido_number} is: {status}'
		except:
			return 'Order number not found. Please check and try again'

	def start_chat_bot(self, bot: ChatBotMedium):
		chat_history_ids = None
		while True:
			input_user = input('You: ')

			if input_user.lower() in self._key_words_finish_chat:
				print('Bot: Goodbye!')
				break

			if any(keyword in input_user.lower() for keyword in self._key_words_status):
				pedido_number = input('Could you please enter your number?')
				answer = self._check_status_pedido(pedido_number)
			else:
				new_user_input_ids = (
					bot.get_tokenizer().
					encode(input_user + bot.get_tokenizer().eos_token, return_tensors='pt'))

				if chat_history_ids is not None:
					bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
				else:
					bot_input_ids = new_user_input_ids

				chat_history_ids = bot.get_model().generate(
					bot_input_ids,
					max_length=1000,
					pad_token_id=bot.get_tokenizer().eos_token_id
				)
				answer = bot.get_tokenizer().decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
				                                    skip_special_tokens=True)

			print(f'Bot: {answer}')
