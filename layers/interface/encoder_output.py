
"""
will define them as a class later 
Paragraph: 
	sentence_list
		word_list
			embedding
			content
			
conversation_history:
	question_list
		word_list
				embedding
				content
	answer_list
		word_list
				embedding
				content
			
current question:
	word_list
			embedding
			content
"""
# maybe it is not a good idea to pass it by class since it is not convenient for original model to process.
# Let us check it later.
class Paragraph:
	def __init__(self, sentence_list):
		self.sentence_list = sentence_list 

class Sentence:
	def __init__(self, word_list):
	    self.word_list = word_list 

class Word:
	def __init__(self, embedding, content):
	    self.embedding = embedding 
	    self.content=content   #content like 'Barry'. String of the word

class ConversationHistory:
	def __init__(self, question_list,answer_list):
		self.question_list = question_list 
		self.answer_list = answer_list 

class CurrentQuestion:
	def __init__(self, word_list ):
		self.word_list = word_list 
