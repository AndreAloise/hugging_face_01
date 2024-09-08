import gradio as gr

from ia_models.faq_question_answer import FaqQuestionAnswer


class FaqQuestionAnswerView:
	def __init__(self, model: FaqQuestionAnswer):
		self.model = model

	def create_gradio_interface(self):
		app = gr.Interface(
			fn=self.model.answering_faq,
			inputs=gr.Dropdown(choices=list(self.model.get_contexts().keys()), label="Select your question"),
			outputs='text',
			title='E-commerce FAQ',
			description='Select a question to get an answer from our FAQ.'
		)
		app.launch(share=False)
