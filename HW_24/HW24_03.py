from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("king + woman - man:", result[0][0])
result = model.most_similar(positive=['India', 'capital'])
print("India + capital:", result[0][0])

result = model.most_similar(positive=['book', 'princesses'], negative=['princess'])
print("book + princesses - princess:", result[0][0])

result = model.most_similar(positive=['blue', 'red'], negative=['green'])
print("colours:", result[0][0])