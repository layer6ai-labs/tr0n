from tr0n.modules.models.model_biggan import Model as ModelBigGAN
from tr0n.modules.models.model_stylegan import Model as ModelStyleGAN
from tr0n.modules.models.model_sgxl import Model as ModelStyleGANXL
from tr0n.modules.models.model_nvae import Model as ModelNVAE

class ModelFactory:
    @staticmethod
    def get_model(config, device, clip_noise_cov=None):
        if config.dec_type == 'biggan':
            return ModelBigGAN(config, device, clip_noise_cov)
        elif config.dec_type == 'stylegan':
            return ModelStyleGAN(config, device, clip_noise_cov)
        elif config.dec_type == 'sgxl':
            return ModelStyleGANXL(config, device, clip_noise_cov)
        elif config.dec_type == 'nvae':
            return ModelNVAE(config, device, clip_noise_cov)
        else:
            raise NotImplementedError
