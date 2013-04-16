# Import the corpus and functions used from nltk library
import nltk
from nltk.corpus import reuters
from nltk.corpus import genesis
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist
from nltk.model import NgramModel
import sys

WIKI_LINE_LIMIT = 10000
N = 3

#adopted from the entropy function
#most appropriate one for nltk
def sentence_probability_modified(s,lm):
	text = nltk.word_tokenize(s)
	p = 1.0
	text = list(lm._lpad) + text + list(lm._rpad)
	for i in range(lm._n-1, len(text)):
		context = tuple(text[i-lm._n+1:i])
		token = text[i]
		p *= lm.prob(token, context)
	return p;

#from language model lecture columbia
#only for 3-gram
def sentence_probability(s,lm):
	s_tokens = nltk.word_tokenize(s)
	n = len(s_tokens)
	assert n>0
	p0 = lm.prob(s_tokens[0],["",""])
	p_total = p0

	if n>1:
		p_total *= lm.prob(s_tokens[1],[s_tokens[0],""])
	elif n==1:
		p1 = lm.prob("",["",s_tokens[n-1]])
		return p0 * p1
	for i in range(2,n-1):
		p_total *= lm.prob(s_tokens[i],[s_tokens[i-2],s_tokens[i-1]])
	
	pn = lm.prob("",[s_tokens[n-2],s_tokens[n-1]])

	p_total *= pn
	return p_total


def sentence_entropy(s,lm):
	s_tokens = nltk.word_tokenize(s)
	return lm.entropy(s_tokens)

def tokenize_file(file_path):
	c=0;
	filename=open(file_path,"r")
	#type(raw) #str
	tokens = []
	for line in filename.readlines():
		if c==WIKI_LINE_LIMIT: break
		tokens+=nltk.word_tokenize(line)
		c +=1
	return tokens


# Tokens contains the words for Genesis and Reuters Trade
tokens = tokenize_file("simple_wikipedia_plaintext.txt")
#tokens = brown.words(categories='news')
#print tokens[1:100]
#tokens = list(genesis.words('english-kjv.txt'))
#tokens.extend(list(reuters.words(categories = 'trade')))
#tokens.extend(list(brown.words(categories='news')))
#tokens.extend(list(reuters.words(categories = 'earn')))

# estimator for smoothing the N-gram model
est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)

# N-gram language model with 3-grams
model = NgramModel(N, tokens, pad_left = True, pad_right = True, estimator=est)
#model = NgramModel(N, tokens, estimator=est)

# Apply the language model to generate 50 words in sequence
#text_words = model.generate(50)

# Concatenate all words generated in a string separating them by a space.
#text = ' '.join([word for word in text_words])

# print the text
#print text

sentence = "This is a sample sentence."
print sentence
print "p:",sentence_probability(sentence,model)
print "p_m:",sentence_probability_modified(sentence,model)
print "en:",sentence_entropy(sentence,model)
#en = model.entropy(sentence)
#print en

sentence1 = "I eat food everyday in the canteen."
print sentence1
p1=sentence_probability(sentence1,model)
pm1=sentence_probability_modified(sentence1,model)
en1=sentence_entropy(sentence1,model)
print "p:", p1
print "p_m:", pm1
print "en:", en1

sentence2 = "food I eat canteen the in everyday."
print sentence2
p2 = sentence_probability(sentence2,model)
pm2 = sentence_probability_modified(sentence2,model)
en2 = sentence_entropy(sentence2,model)
print "p:", p2
print "p_m:", pm2
print "en:", en2

print "sentence_probability winner"
if p1 > p2:
	print sentence1
else:
	print sentence2

print "sentence_probability_modified winner"
if pm1 > pm2:
	print sentence1
else:
	print sentence2

print "sentence_entropy winner"
if en1 > en2:
	print sentence1
else:
	print sentence2

#print model.entropy(sentence1)
#print model.prob("This",["",""]);

#sentence3 = "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced ``no evidence'' that any irregularities took place."
#print sentence3
#print "p:",sentence_probability(sentence3,model)
#print "p_m:",sentence_probability_modified(sentence3,model)
#print "en:",sentence_entropy(sentence3,model)

print "exit to stop"

while(True):

	data = sys.stdin.readline()
	#print data
	if "exit" == data.strip(): break
	print "p:",sentence_probability(data.strip(),model)
	print "p_m:",sentence_probability_modified(data.strip(),model)
	print "en:",sentence_entropy(data.strip(),model)
