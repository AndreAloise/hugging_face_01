import gradio as gr

from ia_models.background_removal import BackgroundRemoval


class BackgroundRemovalView:
	def __init__(self, model: BackgroundRemoval):
		self.model = model

	def create_gradio_interface(self):
		app = gr.Interface(
			fn=self.model.remove_background,
			inputs=gr.components.Image(type="pil"),
			outputs=gr.components.Image(type="pil", format="png"),
			title="Remoção de Background de Imagens",
			description="Envie uma imagem e veja o background sendo removido automaticamente. A imagem resultante será no formato PNG."
		)
		app.launch(share=False)  # True to create a public link
