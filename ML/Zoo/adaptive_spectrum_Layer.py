import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def __init__(
            self,
            input_size: int,
            n_frequencies: int,
            dropout_rate: float
    ):
        super(SimpleGate, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, n_frequencies)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        a gated linear unit with learnable weights.
        :param x: a tensor with shape (B,H,input_size)
        :return: a positive value tensor with shape (B,H,history_size)
        """
        x = self.dropout(x)
        x = self.linear1(x)
        x = x * self.sigmoid(x)
        return x


class Complexifier(nn.Module):
    def __init__(
            self,
            input_dim: int
    ):
        super(Complexifier, self).__init__()
        self.input_dim = input_dim
        self.magnitude_fc = nn.Linear(self.input_dim, 1)
        self.phase_fc = nn.Linear(self.input_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        takes a tensor with shape (B,1,F,input_dim) where input_dim is usually the hidden dim of the ASL and returns a
        complex tensor with the same shape
        :param x:
        :return:
        """
        x = self.relu(x)
        magnitude = self.relu(self.magnitude_fc(x))
        phase = 2 * torch.pi * self.sigmoid(self.phase_fc(x))
        return torch.complex(real=magnitude * torch.cos(phase), imag=magnitude * torch.sin(phase))


class CrossSpectralDensity(nn.Module):
    def __init__(self, n_features, n_frequencies, time_axis):
        super(CrossSpectralDensity, self).__init__()
        self.n_features = n_features
        self.n_frequencies = n_frequencies
        self.time_axis = time_axis

    @staticmethod
    def _batch_xcorr(f, g):
        """
        computes the cross correlation between two tensors with shape (B,T,1))
        :param f: a single feature tensor with shape (B,T,1)
        :param g: yet another single feature tensor with shape (B,T,1)
        :return:
        """
        # USING LOOPS:
        # xcorrs = []
        # for b in range(f.shape[0]):  # looping the samples in the batch
        #     f_b = f[[b]].permute(0, 2, 1)  # shape (1,1,T)
        #     g_b = g[[b]].permute(0, 2, 1)  # shape (1,1,T)
        #     xcorr_b = F.conv1d(f_b, g_b, padding='same')  # shape (1,1,T)
        #     all_xcorrs.append(xcorr_b)
        # xcorrs = torch.cat(all_xcorrs, axis=0)  # shape (B,1,T)
        # xcorrs = all_xcorrs.permute(0, 2, 1)  # (B,T,1)
        # VECTORISED:
        f = f.transpose(1, 2)  # shape (B,1,T)
        g = g.transpose(1, 2)  # shape (B,1,T)
        all_xcorrs = F.conv1d(f, g, padding='same')  # like for i in [B] for j in [B]: compute conv1d(f[i], g[j])
        # we are interested in the case where i=j, so we take the diagonal of the resulting tensor
        diag_indices = torch.arange(all_xcorrs.shape[0])
        xcorrs = all_xcorrs[diag_indices, diag_indices, :].unsqueeze(-1)
        return xcorrs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor with shape (B,H,F), computes the mean cross spectral density of the features (each pair).
        :param x:
        """
        convs = []
        for i in range(self.n_features):
            for j in range(self.n_features):
                f_i, f_j = x[:, :, i].unsqueeze(-1), x[:, :, j].unsqueeze(-1)
                xcorr_ij = self._batch_xcorr(f_i, f_j)
                convs.append(xcorr_ij)
        convs = torch.cat(convs, dim=-1)
        convs_fft = torch.fft.rfft(convs, dim=self.time_axis, norm='ortho')[:, :self.n_frequencies, :]
        mean_convs_fft = convs_fft.mean(dim=-1, keepdim=True)
        return mean_convs_fft


class AdaptiveSpectrumLayer(nn.Module):
    def __init__(
            self,
            input_size: int,
            history_size: int,
            hidden_dim: int,
            sampling_rate: float,
            dropout_rate: float,
            complexify: bool,
            gate: bool,
            use_freqs: bool,
            multidim_fft: bool,
            use_layer_norm: bool,
            per_freq_layer: bool,
            cross_spectrum_density: bool,
            frequency_threshold: float | None = None
    ):
        super(AdaptiveSpectrumLayer, self).__init__()
        self.input_size = input_size
        self.history_size = history_size
        self.hidden_dim = hidden_dim
        self.sampling_rate = sampling_rate
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.complexify = complexify
        self.gate = gate
        self.use_freqs = use_freqs
        self.multidim_fft = multidim_fft
        self.per_freq_layer = per_freq_layer
        self.cross_spectral_density = cross_spectrum_density
        self.fftfreq = None
        self.frequency_threshold = frequency_threshold if frequency_threshold else self.sampling_rate / 2  # Nyquist
        if self.use_freqs:
            if self.multidim_fft:
                self.n_fourier_features = 7  # 2 * 3 + 1
            else:
                self.n_fourier_features = 4  # 3 + 1
        else:
            if self.multidim_fft:
                self.n_fourier_features = 6  # 2 * 3
            else:
                self.n_fourier_features = 3  # 3
        self.time_axis = 1
        self.n_frequencies = self._frequencies_to_use()

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        if self.gate:
            self.gate_of_flattened_ffts = SimpleGate(
                input_size=self.n_frequencies * self.hidden_dim,
                n_frequencies=self.n_frequencies,
                dropout_rate=self.dropout_rate
            )
        else:
            self.gate_of_flattened_ffts = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.n_frequencies * self.hidden_dim, self.n_frequencies)
            )
        fft_projection_backbone = nn.Sequential(
            nn.Linear(self.n_fourier_features, self.hidden_dim),
            self.relu,
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # self.relu
        )
        if self.per_freq_layer:
            self.fft_projection = nn.ModuleList([
                # for each frequency, we project features from fourier (like magnitude, angle,...)
                fft_projection_backbone for _ in range(self.n_frequencies)
            ])
        else:
            self.fft_projection = fft_projection_backbone

        if self.complexify:
            self.complexifier_layers = nn.ModuleList([
                # convert a hidden representation to a complex number
                Complexifier(self.hidden_dim) for _ in range(self.n_frequencies)
            ])
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.input_size)
        if self.cross_spectral_density:
            self.csd = CrossSpectralDensity(self.input_size, self.n_frequencies, self.time_axis)

        self.fft_weights = None  # saving last fft weights for visualization
        self.fft_magnitude = None # saving last fft magnitude value for visualization

    def _frequencies_to_use(self) -> int:
        """
        we trim the fftfreq values given the frequency threshold and return the number of frequencies to consider,
        :return: number of frequencies to consider, that represent frequencies from 0 to frequency_threshold
        """
        frequencies = torch.fft.rfftfreq(self.history_size, 1 / self.sampling_rate)
        frequencies_mask = frequencies <= self.frequency_threshold
        self.fftfreq = frequencies[frequencies_mask]
        n_frequencies = frequencies_mask.sum().item()
        return n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        applies rfft on the input and learns weights for the frequencies. then applies irfft on the weighted fft.
        :param x: a tensor with shape (B,H,F)
        :return: a tensor with shape (B,H,F)
        """
        fft = torch.fft.rfft(x, dim=self.time_axis, norm='ortho')[:, :self.n_frequencies, :]  # (B,n_freqs,F)
        if self.cross_spectral_density:
            csd = self.csd(x)
            fft = fft + csd
        magnitude, angle = torch.abs(fft), torch.angle(fft)  # (B,n_freqs,F), (B,n_freqs,F)
        self.fft_magnitude = magnitude

        fourier_features = torch.stack([magnitude, torch.sin(angle), torch.cos(angle)], dim=-1)  # (B,n_freqs,F,3)
        if self.multidim_fft:
            fftn = torch.fft.rfftn(x, dim=self.time_axis)[:, :self.n_frequencies, :]  # (B,n_freqs,F)
            magnitude_n, angle_n = torch.abs(fftn), torch.angle(fftn)  # (B,n_freqs,F), (B,n_freqs,F)
            fourier_features_n = torch.stack([magnitude_n, torch.sin(angle_n), torch.cos(angle_n)],
                                             dim=-1)  # (B,n_freqs,F,3)
            fourier_features = torch.cat([fourier_features, fourier_features_n], dim=-1)  # (B,n_freqs,F,6)
        if self.use_freqs:
            fourier_features = torch.cat([
                fourier_features,
                self.fftfreq.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(fft.shape[0], 1, fft.shape[-1], 1).to(
                    fft.device)
            ], dim=-1)
        if self.per_freq_layer:
            projected_fourier_features = [
                self.fft_projection[i](fourier_features[:, [i], :, :])
                for i in range(self.n_frequencies)
            ]  # [(B,1,F,hidden_dim)_{1}, ...,(B,1,F,hidden_dim)_{n_freqs}]
            flattened_fourier_features = torch.cat(projected_fourier_features, dim=-1)  # (B,1,F,hidden_dim*n_freqs)
        else:
            flattened_fourier_features = self.fft_projection(fourier_features)  # (B,n_freqs,F,hidden_dim)
            flattened_fourier_features = einops.rearrange(flattened_fourier_features, 'b n f h -> b 1 f (h n)')

        weights = self.sigmoid(self.gate_of_flattened_ffts(flattened_fourier_features))  # (B,1,F,n_frequencies)
        if self.complexify:
            new_fft = torch.cat([
                self.complexifier_layers[i](projected_fourier_features[i]) for i in range(self.n_frequencies)
            ], dim=1).squeeze(-1)
            # weighted_Fft is a convex sum of fft and new_Fft:
            w = weights.squeeze(1).permute(0, 2, 1)
            weighted_fft = w * new_fft + (1 - w) * fft  # (B,n_frequencies,F)
        else:
            weighted_fft = fft * weights.squeeze(1).permute(0, 2, 1)  # (B,n_frequencies,F)
        self.fft_weights = weights.squeeze(1).permute(0, 2, 1)
        weighted_fft = torch.nn.functional.pad(
            input=weighted_fft,
            pad=(0, 0, 0, (self.history_size // 2 + 1) - self.n_frequencies, 0, 0)
        )
        reconstructed_x = torch.fft.irfft(weighted_fft, dim=self.time_axis, norm='ortho')  # (B,H,F)
        if self.use_layer_norm:
            reconstructed_x = self.layer_norm(reconstructed_x + x)
        return reconstructed_x


if __name__ == '__main__':
    T = 0.1024
    fs = 5000
    t = torch.arange(0, T, 1 / fs)
    x = torch.stack(
        [
            torch.sin(2 * np.pi * 50 * t),
            torch.sin(2 * np.pi * 120 * t),
            torch.sin(2 * np.pi * 30 * t) + torch.sin(2 * np.pi * 9 * t),
        ],
        dim=1
    ).repeat(4, 1, 1)
    asl = AdaptiveSpectrumLayer(
        history_size=x.shape[1],
        hidden_dim=10,
        sampling_rate=fs,
        frequency_threshold=200
    )
    y = asl(x)
    print(y.shape)
