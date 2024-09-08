from transformers import pipeline


class FaqQuestionAnswer:
	def __init__(self, contexts):
		self.pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')
		self._contexts = contexts

	def get_contexts(self):
		return self._contexts

	def _get_answer(self, question, context):
		return self.pipeline(question=question, context=context)

	def answering_faq(self, question):
		context = self.get_contexts()[question]
		answer = self._get_answer(question, context)
		return answer.get('answer')
