import torch
import torch.nn as nn
from LTSFLinear.layers.Embed import (
	DataEmbedding,
	DataEmbedding_wo_pos,
	DataEmbedding_wo_temp,
	DataEmbedding_wo_pos_temp
)
from LTSFLinear.layers.AutoCorrelation import (
	AutoCorrelation,
	AutoCorrelationLayer
)
from LTSFLinear.layers.Autoformer_EncDec import (
	Encoder,
	Decoder,
	EncoderLayer,
	DecoderLayer,
	my_Layernorm,
	series_decomp
)


class Autoformer(nn.Module):
	"""
	Autoformer is the first method to achieve the series-wise connection,
	with inherent O(LlogL) complexity
	"""

	def __init__(self, target_lags, label_len, output_attention, enc_in, d_model, dropout, dec_in, embed_type, factor,
				 d_ff, e_layers, activation, n_heads, d_layers, c_out, moving_avg, output_size, embed=None, freq=None):
		super(Autoformer, self).__init__()
		self.target_lags = target_lags
		self.label_len = label_len
		self.output_attention = output_attention
		self.enc_in = enc_in
		self.d_model = d_model
		self.embed = embed
		self.freq = freq
		self.dropout = dropout
		self.dec_in = dec_in
		self.embed_type = embed_type
		self.factor = factor
		self.d_ff = d_ff
		self.e_layers = e_layers
		self.activation = activation
		self.n_heads = n_heads
		self.d_layers = d_layers
		self.c_out = c_out
		self.moving_avg = moving_avg
		self.output_size =output_size

		# Decomp
		kernel_size = self.moving_avg
		self.decomp = series_decomp(kernel_size)

		# Embedding
		# The series-wise connection inherently contains the sequential information.
		# Thus, we can discard the position embedding of transformers.
		if self.embed_type == 0:
			self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
			self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)
		elif self.embed_type == 1:
			self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
			self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)
		elif self.embed_type == 2:
			self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
			self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)

		elif self.embed_type == 3:
			self.enc_embedding = DataEmbedding_wo_temp(self.enc_in, self.d_model, self.dropout)
			self.dec_embedding = DataEmbedding_wo_temp(self.dec_in, self.d_model, self.dropout)
		elif self.embed_type == 4:
			self.enc_embedding = DataEmbedding_wo_pos_temp(self.enc_in, self.d_model, self.dropout)
			self.dec_embedding = DataEmbedding_wo_pos_temp(self.dec_in, self.d_model, self.dropout)

		# Encoder
		self.encoder = Encoder(
			[
				EncoderLayer(
					AutoCorrelationLayer(
						AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
										output_attention=self.output_attention),
						self.d_model, self.n_heads),
					self.d_model,
					self.d_ff,
					moving_avg=self.moving_avg,
					dropout=self.dropout,
					activation=self.activation
				) for l in range(self.e_layers)
			],
			norm_layer=my_Layernorm(self.d_model)
		)
		# Decoder
		self.decoder = Decoder(
			[
				DecoderLayer(
					AutoCorrelationLayer(
						AutoCorrelation(True, self.factor, attention_dropout=self.dropout,
										output_attention=False),
						self.d_model, self.n_heads),
					AutoCorrelationLayer(
						AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
										output_attention=False),
						self.d_model, self.n_heads),
					self.d_model,
					self.c_out,
					self.d_ff,
					moving_avg=self.moving_avg,
					dropout=self.dropout,
					activation=self.activation,
				)
				for l in range(self.d_layers)
			],
			norm_layer=my_Layernorm(self.d_model),
			projection=nn.Linear(self.d_model, self.output_size, bias=True),
			trend_projection=nn.Linear(self.enc_in,self.output_size,bias=False)
		)


	def __str__(self):
		return f"LTSFAutoFormer[{self.target_lags}, {self.output_attention}, {self.enc_in}, {self.d_model}, " \
			   f"{self.embed}, {self.freq}, {self.dropout}, {self.dec_in}, '{self.embed_type}', {self.factor}, " \
			   f"{self.d_ff}, {self.e_layers}, '{self.activation}', {self.n_heads}, {self.d_layers}, {self.c_out}, {self.moving_avg}]"

	def forward(self, x_enc,
				x_mark_enc=None, x_mark_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		# decomp init
		mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.target_lags, 1)
		zeros = torch.zeros([x_enc.shape[0], self.target_lags, x_enc.shape[-1]], device=x_enc.device)
		seasonal_init, trend_init = self.decomp(x_enc)
		# decoder input
		trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
		seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
		# enc
		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
		# dec
		dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
		seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
												 trend=trend_init)
		# final
		dec_out = trend_part + seasonal_part

		if self.output_attention:
			return dec_out[:, -self.target_lags:, :], attns
		else:
			return dec_out[:, -self.target_lags:, :]  # [B, L, D]
