import torch
import torch.nn as nn
from LTSFLinear.layers.Transformer_EncDec import (
	Decoder,
	DecoderLayer,
	Encoder,
	EncoderLayer
)
from LTSFLinear.layers.SelfAttention_Family import (
	FullAttention,
	AttentionLayer
)
from LTSFLinear.layers.Embed import (
	DataEmbedding,
	DataEmbedding_wo_pos,
	DataEmbedding_wo_temp,
	DataEmbedding_wo_pos_temp
)
from ML.Zoo.adaptive_spectrum_Layer import AdaptiveSpectrumLayer


class Transformer(nn.Module):
	"""
	Vanilla Transformer with O(L^2) complexity
	"""

	def __init__(self,feature_lags, target_lags, label_len, output_attention, enc_in, d_model, dropout, dec_in, embed_type, factor,
				 d_ff, e_layers, activation, n_heads, d_layers, c_out, embed=None, freq=None):
		super(Transformer, self).__init__()
		self.feature_lags = feature_lags
		self.pred_len = target_lags
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

		# Embedding
		if self.embed_type == 0:
			self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
			self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)
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
					AttentionLayer(
						FullAttention(
							False, self.factor, attention_dropout=self.dropout,
							output_attention=self.output_attention
						),
						self.d_model, self.n_heads
					),
					self.d_model,
					self.d_ff,
					dropout=self.dropout,
					activation=self.activation
				) for l in range(self.e_layers)
			],
			norm_layer=torch.nn.LayerNorm(self.d_model)
		)
		# Decoder
		self.decoder = Decoder(
			[
				DecoderLayer(
					AttentionLayer(
						FullAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
						self.d_model, self.n_heads),
					AttentionLayer(
						FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
						self.d_model, self.n_heads),
					self.d_model,
					self.d_ff,
					dropout=self.dropout,
					activation=self.activation,
				)
				for l in range(self.d_layers)
			],
			norm_layer=torch.nn.LayerNorm(self.d_model),
			projection=nn.Linear(self.d_model, self.c_out, bias=True)
		)
		self.use_adl = False
		if self.use_adl:
			self.adl1 = AdaptiveSpectrumLayer(
				history_size=self.feature_lags,
				hidden_dim=self.d_model,
				sampling_rate=5000,
				frequency_threshold=200,
				complexify=False,
				gate=True,
				use_freqs=False,
				dropout_rate=0.1,
				multidim_fft=True
			)

	def __str__(self):
		return f"LTSFVanillaTransformer[{self.pred_len}, {self.output_attention}, {self.enc_in}, {self.d_model}, " \
			   f"{self.embed}, {self.freq}, {self.dropout}, {self.dec_in}, '{self.embed_type}', {self.factor}, " \
			   f"{self.d_ff}, {self.e_layers}, '{self.activation}', {self.n_heads}, {self.d_layers}, {self.c_out}]"

	def forward(self, x_enc,
				x_mark_enc=None, x_mark_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		x_dec = torch.zeros(x_enc.shape[0], self.pred_len, self.c_out).to(x_enc.device)
		# x_dec = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
		# x_dec = torch.cat([batch_y[:, :self.args.label_len, :], x_dec], dim=1).float().to(self.device)

		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		if self.use_adl:
			enc_out = self.adl1(enc_out)
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		dec_out = self.dec_embedding(x_dec, x_mark_dec)
		dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

		if self.output_attention:
			return dec_out[:, -self.pred_len:, :], attns
		else:
			return dec_out[:, -self.pred_len:, :]  # [B, L, D]
