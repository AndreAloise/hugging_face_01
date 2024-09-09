import gradio as gr

from services.chat_bot_medium_service import ChatBotMediumService


class ChatBotMediumView:
	def __init__(self, service: ChatBotMediumService):
		self.service = service

	def create_gradio_interface(self):
		with gr.Blocks() as app:
			chatbot = gr.Chatbot()
			msg = gr.Textbox(placeholder='Type your message here...')

			state = gr.State(None)
			waiting_number_pedido = gr.State(False)

			msg.submit(
				self.service.entry_process_for_view,
				[msg, chatbot, state, waiting_number_pedido],
				[chatbot, state, waiting_number_pedido, msg]
			)

		app.launch(share=False)
