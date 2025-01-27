import torch
import torch.nn as nn
import torch.nn.functional as F
from LTSFLinear.FEDformer.layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp, \
	DataEmbedding_wo_temp
from LTSFLinear.FEDformer.layers.AutoCorrelation import AutoCorrelationLayer
from LTSFLinear.FEDformer.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
# from LTSFLinear.FEDformer.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from LTSFLinear.FEDformer.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, \
	series_decomp, series_decomp_multi


class FedFormer(nn.Module):
	"""
	FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
	"""

	def __init__(self,
				 feature_lags,
				 label_len,
				 target_lags,
				 output_attention,
				 enc_in,
				 d_model,
				 dropout,
				 dec_in,
				 n_heads,
				 e_layers,
				 d_layers,
				 activation,
				 moving_avg,
				 embed_type,
				 d_ff,
				 c_out,
				 version,
				 mode_select,
				 modes,
				 output_size
				 ):
		super(FedFormer, self).__init__()
		self.version = version
		self.mode_select = mode_select
		self.modes = modes
		self.feature_lags = feature_lags
		self.label_len = label_len
		self.pred_len = target_lags
		self.output_attention = output_attention
		self.moving_avg = moving_avg
		self.enc_in = enc_in
		self.dec_in = dec_in
		self.d_model = d_model
		self.dropout = dropout
		self.embed_type = embed_type
		self.freq = None
		self.n_heads = n_heads
		self.d_ff = d_ff
		self.e_layers = e_layers
		self.d_layers = d_layers
		self.c_out = c_out
		self.activation = activation
		self.embed = None
		self.output_size = output_size

		# Decomp
		kernel_size = self.moving_avg
		if isinstance(kernel_size, list):
			self.decomp = series_decomp_multi(kernel_size)
		else:
			self.decomp = series_decomp(kernel_size)
		if self.embed_type == 0:
			self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
													  self.dropout)
			self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
													  self.dropout)
		elif self.embed_type == 1:
			self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
											   self.dropout)
			self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
											   self.dropout)
		elif self.embed_type == 2:
			self.enc_embedding = DataEmbedding_wo_pos_temp(self.enc_in, self.d_model, self.embed, self.freq,
														   self.dropout)
			self.dec_embedding = DataEmbedding_wo_pos_temp(self.dec_in, self.d_model, self.embed, self.freq,
														   self.dropout)
		elif self.embed_type == 3:
			self.enc_embedding = DataEmbedding_wo_temp(self.enc_in, self.d_model, self.embed, self.freq,
													   self.dropout)
			self.dec_embedding = DataEmbedding_wo_temp(self.dec_in, self.d_model, self.embed, self.freq,
													   self.dropout)

		if self.version == 'Wavelets':
			pass
			# encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=self.L, base=self.base)
			# decoder_self_att = MultiWaveletTransform(ich=self.d_model, L=self.L, base=self.base)
			# decoder_cross_att = MultiWaveletCross(in_channels=self.d_model,
			# 									  out_channels=self.d_model,
			# 									  seq_len_q=self.feature_lags // 2 + self.pred_len,
			# 									  seq_len_kv=self.feature_lags,
			# 									  modes=self.modes,
			# 									  ich=self.d_model,
			# 									  base=self.base,
			# 									  activation=self.cross_activation)
		else:
			encoder_self_att = FourierBlock(in_channels=self.d_model,
											out_channels=self.d_model,
											seq_len=self.feature_lags,
											modes=self.modes,
											mode_select_method=self.mode_select)
			decoder_self_att = FourierBlock(in_channels=self.d_model,
											out_channels=self.d_model,
											seq_len=self.feature_lags // 2 + self.pred_len,
											modes=self.modes,
											mode_select_method=self.mode_select)
			decoder_cross_att = FourierCrossAttention(in_channels=self.d_model,
													  out_channels=self.d_model,
													  seq_len_q=self.feature_lags // 2 + self.pred_len,
													  seq_len_kv=self.feature_lags,
													  modes=self.modes,
													  mode_select_method=self.mode_select)
		# Encoder
		enc_modes = int(min(self.modes, self.feature_lags // 2))
		dec_modes = int(min(self.modes, (self.feature_lags // 2 + self.pred_len) // 2))
		print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

		self.encoder = Encoder(
			[
				EncoderLayer(
					AutoCorrelationLayer(
						encoder_self_att,
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
						decoder_self_att,
						self.d_model, self.n_heads),
					AutoCorrelationLayer(
						decoder_cross_att,
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
			trend_projection=nn.Linear(self.enc_in, self.output_size, bias=False)
		)

	def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
				enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		# decomp init
		mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
		seasonal_init, trend_init = self.decomp(x_enc)
		# decoder input
		trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
		seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
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
			return dec_out[:, -self.pred_len:, :], attns
		else:
			return dec_out[:, -self.pred_len:, :]  # [B, L, D]

	def __str__(self):
		return f"LTSFFedFormer[{self.feature_lags}, {self.pred_len}, {self.output_attention}, {self.enc_in}, {self.d_model}, " \
			   f"{self.embed}, {self.freq}, {self.dropout}, {self.dec_in}, '{self.embed_type}',  " \
			   f"{self.d_ff}, {self.e_layers}, '{self.activation}', {self.n_heads}, {self.d_layers}, {self.c_out}, {self.moving_avg}, {self.version}, {self.mode_select}, {self.modes}]"


if __name__ == '__main__':
	d = dict(
		modes=32,
		mode_select='random',
		version='Fourier',
		moving_avg=24,
		feature_lags=96,
		label_len=48,
		target_lags=96,
		output_attention=True,
		enc_in=7,
		dec_in=7,
		d_model=16,
		dropout=0.05,
		# freq='h',
		n_heads=8,
		d_ff=16,
		e_layers=2,
		d_layers=1,
		c_out=7,
		activation='gelu',
		embed_type=3
	)

	model = FedFormer(**d)

	print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
	enc = torch.randn([3, d['feature_lags'], 7])
	enc_mark = torch.randn([3, d['feature_lags'], 4])

	dec = torch.randn([3, d['feature_lags'] // 2 + d['target_lags'], 7])
	dec_mark = torch.randn([3, d['feature_lags'] // 2 + d['target_lags'], 4])
	out = model.forward(enc, enc_mark, dec, dec_mark)
	print(out)
