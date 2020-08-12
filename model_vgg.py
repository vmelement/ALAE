# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import losses
from net import *
import numpy as np

class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)

def L2Norm_Instance(x):
    return F.normalize(x.view(x.size(0), -1), dim=1, p=2).view(x.size())

class ArcFace(nn.Module):

    def __init__(self, latent_size, identity_count, loss_param, gain=np.sqrt(2.0), lrmul=1.0):
        super(ArcFace, self).__init__()

        self.latent_size = latent_size
        self.identity_count = identity_count
        self.loss_param = loss_param
        self.gain = gain
        self.lrmul = lrmul
        self.fc_weight = Parameter(torch.Tensor(identity_count, latent_size))
        self.reset_parameters()
        print(self.fc_weight, self.fc_weight.shape)

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.latent_size) * self.lrmul
        init.normal_(self.fc_weight, mean=0, std=self.std / self.lrmul)
        setattr(self.fc_weight, 'lr_equalization_coef', self.std)

    # lfj
    def forward(self, embedding, y, lod, blend_factor, d_train, ae):
        # embedding: (64, 1, 512), y: (64,)
        # self.fc_weight =L2Norm_Instance(self.fc_weight)

        embedding_normalized = L2Norm_Instance(embedding) * self.loss_param['embedding_norm']
        # print(embedding_normalized, embedding_normalized.shape)
        fc = F.linear(embedding_normalized, self.fc_weight)
        print(fc, fc.shape)
        exit(0)

        pass

class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="",
                 encoder="", z_regression=False, identity_count=0, arcface_start_lod=4):
        super(Model, self).__init__()

        self.layer_count = layer_count
        self.z_regression = z_regression
        self.arcface_start_lod = arcface_start_lod

        self.mapping_tl = MAPPINGS["MappingToLatent"](
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.mapping_fl = MAPPINGS["MappingFromLatent"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.encoder = ENCODERS[encoder](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        loss_param={'embedding_norm': 64.0, 'm1': 1.0, 'm2': 0.38, 'm3': 0.0}
        self.arcface = ArcFace(latent_size, identity_count, loss_param) if identity_count>0 else None

        self.dlatent_avg = DLatent(latent_size, self.mapping_fl.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping_fl(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_fl.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping_fl(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_fl.num_layers, 1)

                layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if (self.truncation_psi is not None) and not no_truncation:
            layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):
        Z = self.encoder(x, lod, blend_factor)
        Z_ = self.mapping_tl(Z)
        return Z[:, :1], Z_[:, 1, 0]

    def forward(self, x, y, lod, blend_factor, d_train, ae):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            s, rec = self.generate(lod, blend_factor, z=z, mixing=False, noise=True, return_styles=True)

            Z, d_result_real = self.encode(rec, lod, blend_factor)

            assert Z.shape == s.shape

            if self.z_regression:
                Lae = torch.mean(((Z[:, 0] - z)**2))
            else:
                Lae = torch.mean(((Z - s.detach())**2))

            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True)

            self.encoder.requires_grad_(True)

            embedding_real, d_result_real = self.encode(x, lod, blend_factor)

            _, d_result_fake = self.encode(Xp.detach(), lod, blend_factor)

            loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)

            # compute loss_arcface, enabled only when image resolution is large enough
            # lfj
            if lod>=self.arcface_start_lod:
                # loss_arc =
                pass

            return loss_d
        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)

            self.encoder.requires_grad_(False)

            rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), noise=True)

            _, d_result_fake = self.encode(rec, lod, blend_factor)

            loss_g = losses.generator_logistic_non_saturating(d_result_fake)

            return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters()) + list(self.mapping_fl.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)


class GenModel(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="", encoder="", z_regression=False):
        super(GenModel, self).__init__()

        self.layer_count = layer_count

        self.mapping_fl = MAPPINGS["MappingFromLatent"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_fl.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None):
        styles = self.mapping_fl(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_fl.num_layers, 1)

        layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
        styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, True)
        return rec

    def forward(self, x):
        return self.generate(self.layer_count-1, 1.0, z=x)
