from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from DecoderRNN import DecoderRNN
from EncoderRNN import EncoderRNN
from utils import *

#Training the Model
"""
To train we run the input sentence through the encoder,
and keep track of every output and the latest hidden state.
Then the decoder is given the <SOS> token as its first input,
and the last hidden state of the encoder as its first hidden state.

"Teacher forcing" is the concept of using the real target outputs 
as each next input, instead of using the decoder’s guess as the next input. 
Using teacher forcing causes it to converge faster 
but when the trained network is exploited, it may exhibit instability.

You can observe outputs of teacher-forced networks that
read with coherent grammar but wander far from the correct translation
- intuitively it has learned to represent the output grammar and can "pick up"
the meaning once the teacher tells it the first few words, 
but it has not properly learned how to create the sentence
from the translation in the first place.

Because of the freedom PyTorch’s autograd gives us, 
we can randomly choose to use teacher forcing or not with a simple if statement. 
Turn teacher_forcing_ratio up to use more of it.
"""

teacher_forcing_ratio = 0.5
MAX_LENGTH = 10


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
	# 							  not target_length, it's MAX_length!
	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
		#print(encoder_output)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)
	decoder_hidden = encoder_hidden # decoder's first hidden vector is encoder's hidden!

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			loss += criterion(decoder_output, target_tensor[di]) #criterion = nn.NLLLoss()
			decoder_input = target_tensor[di]  # Teacher forcing

	else:
		# Without teacher forcing: use its own predictions as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()  # detach from history as input

			loss += criterion(decoder_output, target_tensor[di]) #criterion = nn.NLLLoss()
			if decoder_input.item() == EOS_token:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
	"""
	Start a timer
	Initialize optimizers and criterion
	Create set of training pairs
	Start empty losses array for plotting    
	"""
	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	training_pairs = [tensorsFromPair(random.choice(pairs))
					  for i in range(n_iters)]
	criterion = nn.NLLLoss()

	for iter in range(1, n_iters + 1):
		training_pair = training_pairs[iter - 1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train(input_tensor, target_tensor, encoder,
					 decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
										 iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

	showPlot(plot_losses)




def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
	with torch.no_grad():
		input_tensor = tensorFromSentence(input_lang, sentence)
		input_length = input_tensor.size()[0]
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei],
													 encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]

		decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

		decoder_hidden = encoder_hidden

		decoded_words = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for di in range(max_length):
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden)#, encoder_outputs)
			#decoder_output, decoder_hidden, decoder_attention = decoder(
			#	decoder_input, decoder_hidden)#, encoder_outputs)
			#decoder_attentions[di] = decoder_attention.data
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(output_lang.index2word[topi.item()])

			decoder_input = topi.squeeze().detach()

		return decoded_words#, decoder_attentions[:di + 1]



def evaluateRandomly(encoder, decoder, n=10):
	for i in range(n):
		pair = random.choice(pairs)
		print('original1: ', pair[0])
		print('original2: ', pair[1])
		output_words = evaluate(encoder, decoder, pair[0])
		#output_words, attentions = evaluate(encoder, decoder, pair[0])
		output_sentence = ' '.join(output_words)
		print('our translation: ', output_sentence)
		print('')



hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device) # Notice input_lang.n_words
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)# Notice output_lang.n_words

trainIters(encoder1, decoder1, 75000, print_every=5000)
evaluateRandomly(encoder1, decoder1)
