from PIL import Image
from transformers import pipeline


class BackgroundRemoval:
	def __init__(self, image_path: str):
		self.pipe = (
			pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True))
		self.pillow_mask = self.pipe(image_path, return_mask=True)
		self.pillow_image = self.pipe(image_path)

	def get_pillow_image(self):
		return self.pillow_image

	def show_image(self):
		"""
			Opens the clean image in a pop-up
		"""
		image = self.get_pillow_image()
		if isinstance(image, Image.Image):
			image.show()
		else:
			print("A imagem retornada não é uma instância de PIL.Image.")
