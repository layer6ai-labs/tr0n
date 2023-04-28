from tr0n.modules.datasets.transforms import init_transform_dict
from tr0n.modules.datasets.coco_dataset import CocoCaptions
from tr0n.modules.datasets.random_gen_dataset import RandomBigGANDataset, RandomStyleGANDataset, RandomNVAEDataset, RandomStyleGANXLDataset
from tr0n.modules.datasets.synthetic_pair_dataset import SyntheticBigGANDataset, SyntheticStyleGANDataset, SyntheticNVAEDataset
from tr0n.modules.datasets.text_dataset import TextDataset
from tr0n.modules.datasets.image_dataset import ImageDataset
from torch.utils.data import DataLoader

class DataFactory:
    @staticmethod
    def get_loader(config, train=False, codebook=None, mapper=None):
        if config.dec_type == 'biggan':
            gen_image_size = config.biggan_resolution
        elif config.dec_type == 'stylegan':
            gen_image_size = int(config.stylegan_gen.split('-')[-1])
        elif config.dec_type == 'sgxl':
            gen_image_size = int(config.stylegan_gen.split('-')[-1])
        elif config.dec_type == 'nvae':
            assert config.nvae_model == 'celeba_256' or config.nvae_model == 'ffhq'
            gen_image_size = 256
        else:
            raise NotImplementedError
        transform_dict = init_transform_dict(config.clip_image_size, gen_image_size)

        if config.dataset_name == 'random_gen_biggan':
            assert config.dec_type == 'biggan'
            # sampling from GAN space of BigGAN
            dataset = RandomBigGANDataset(
                            dataset_size = config.datasize_syn,
                            y_codebook = codebook,
                            n_classes = config.n_classes,
                            dim_z = config.dim_z,
                            dim_y = config.shared_dim,
                            z_thresh = config.z_thresh,
                            z_dist_type = config.z_dist_type, # [truncate|normal|uniform]
                            y_dist_type = config.y_dist_type, # [categorical|normal]
                            z_truncation = config.z_truncation)
            return DataLoader(dataset, batch_size=config.batch_size,
                    num_workers=config.num_workers)
        
        elif config.dataset_name == 'random_gen_stylegan':
            assert config.dec_type == 'stylegan'
            # sampling from GAN space of StyleGAN
            dataset = RandomStyleGANDataset(
                            dataset_size = config.datasize_syn,
                            w_mapper = mapper,
                            dim_z = config.dim_z,
                            z_dist_type = config.z_dist_type, # [truncate|normal|uniform]
                            z_truncation = config.z_truncation)
            return DataLoader(dataset, batch_size=config.batch_size,
                    num_workers=config.num_workers)
        
        elif config.dataset_name == 'random_gen_sgxl':
            assert config.dec_type == 'sgxl'
            # sampling from GAN space of StyleGANXL
            # the class embedding should always be 512, same as w_space
            dataset = RandomStyleGANXLDataset(
                            dataset_size = config.datasize_syn,
                            w_mapper = mapper,
                            w_avg_codebook = codebook,
                            dim_z = config.dim_z,
                            dim_y = 512,
                            n_classes = 1000,
                            z_dist_type = config.z_dist_type, # [truncate|normal|uniform]
                            z_truncation = config.z_truncation,
                            y_dist_type = config.y_dist_type
                            )
            return DataLoader(dataset, batch_size=config.batch_size,
                    num_workers=config.num_workers)
         
        elif config.dataset_name == 'random_gen_nvae':
            assert config.dec_type == 'nvae'
            # sampling from latent space of NVAE
            dataset = RandomNVAEDataset(
                            dataset_size = config.datasize_syn,
                            dim_z = config.dim_z,
                            z_truncation = config.z_truncation)
            return DataLoader(dataset, batch_size=config.batch_size,
                    num_workers=config.num_workers)
        
        elif config.dataset_name == 'synthetic_biggan':
            assert config.dec_type == 'biggan'
            # using synthetic data generated from biggan to train the translator net
            if train:
                dataset = SyntheticBigGANDataset(synthetic_latents_path = config.synthetic_latents_path)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
            else:
                transform = transform_dict["val"]
                dataset = CocoCaptions(root=config.val_image_dir,
                                       annFile=config.val_ann_file,
                                       transform=transform,
                                       train=False)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)
        
        elif config.dataset_name == 'synthetic_stylegan':
            assert config.dec_type == 'stylegan'
            # using synthetic data generated from stylegan to train the translator net
            if train:
                dataset = SyntheticStyleGANDataset(synthetic_latents_path = config.synthetic_latents_path)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
            else:
                dataset = TextDataset(config.dec_type, stylegan_eval_mode=config.stylegan_eval_mode)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)
         
        elif config.dataset_name == 'synthetic_nvae':
            assert config.dec_type == 'nvae'
            # using synthetic data generated from nvae to train the translator net
            if train:
                dataset = SyntheticNVAEDataset(synthetic_latents_path = config.synthetic_latents_path)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
            else:
                dataset = TextDataset(config.dec_type, nvae_model=config.nvae_model)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'text':
            assert not train
            dataset = TextDataset(config.dec_type, stylegan_eval_mode=config.stylegan_eval_mode, nvae_model=config.nvae_model)
            return DataLoader(dataset, batch_size=config.batch_size,
                    shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'image':
            assert not train
            transform = transform_dict["val"]
            transform_orig = transform_dict["test"]
            dataset = ImageDataset(root=config.val_image_dir, transform=transform, transform_orig=transform_orig)
            return DataLoader(dataset, batch_size=config.batch_size,
                    shuffle=False, num_workers=config.num_workers)
        
        else:
            raise NotImplementedError
