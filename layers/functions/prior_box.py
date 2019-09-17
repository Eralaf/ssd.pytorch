from __future__ import division

from math       import sqrt     as sqrt
from itertools  import product  as product

import torch


class PriorBox(object):

    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()

        self.image_size    = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors    = len(cfg['aspect_ratios'])
        self.variance      = cfg['variance'] or [0.1]
        self.feature_maps  = cfg['feature_maps']
        self.min_sizes     = cfg['min_sizes']
        self.max_sizes     = cfg['max_sizes']
        self.steps         = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios'] # Utilisation des aspects
                                                  # ratios, voir plus bas pour
                                                  # la gestion des 1/3, 1/2
        self.clip          = cfg['clip']
        self.version       = cfg['name']

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):

            # k -> indice de la feature map
            # f -> donne la dimension des côtés de la feature map ie : 38, 19,...

            for i, j in product(range(f), repeat=2):

                # i,j correspondent aux coordonnées d'une feature map locations
                # ici cette boucle parcours toutes les features map locations

                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    # C'est pour ça que l'on a que les aspects ratios 2 et/ou 3, le 1/3 et 1/2 est géré ici
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0) # bbox ne peux pas faire plus de 100% de l'image donc ok
                                        # et 0 < cx < 1
                                        #    0 < cy < 1
        return output
