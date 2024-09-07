from PIL import Image
from transformers import pipeline


class BackgroundRemoval:
	def __init__(self):
		self.pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
		self.pillow_mask = ''

	def remove_background(self, image):
		self.pillow_mask = self.pipe(image, return_mask=True)

		# Aplicar máscara na imagem original
		pillow_image = self.pipe(image)

		return pillow_image

	def show_image(self, image):
		"""
			Opens the clean image in a pop-up
		"""
		image = self.remove_background(image)
		if isinstance(image, Image.Image):
			image.show()
		else:
			print("A imagem retornada não é uma instância de PIL.Image.")
