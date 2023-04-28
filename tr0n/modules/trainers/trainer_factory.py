from tr0n.modules.trainers.trainer_synthetic_biggan import Trainer as TrainerBigGan
from tr0n.modules.trainers.trainer_synthetic_stylegan import Trainer as TrainerStyleGan
from tr0n.modules.trainers.trainer_synthetic_nvae import Trainer as TrainerNVae
from tr0n.modules.trainers.hybrid_evaluator_synthetic_biggan import Evaluator as HybridEvaluatorBigGan
from tr0n.modules.trainers.text_hybrid_evaluator_synthetic_biggan import Evaluator as TextHybridEvaluatorBigGan
from tr0n.modules.trainers.text_hybrid_evaluator_synthetic_stylegan import Evaluator as TextHybridEvaluatorStyleGan
from tr0n.modules.trainers.text_hybrid_evaluator_synthetic_nvae import Evaluator as TextHybridEvaluatorNVae
from tr0n.modules.trainers.image_hybrid_evaluator_synthetic_biggan import Evaluator as ImageHybridEvaluatorBigGan
from tr0n.modules.trainers.image_hybrid_evaluator_synthetic_stylegan import Evaluator as ImageHybridEvaluatorStyleGan
from tr0n.modules.trainers.interp_image_hybrid_evaluator_synthetic_biggan import Evaluator as InterpImageHybridEvaluatorBigGan
from tr0n.modules.trainers.interp_image_hybrid_evaluator_synthetic_stylegan import Evaluator as InterpImageHybridEvaluatorStyleGan

class TrainerFactory:
    @staticmethod    
    def get_trainer(*args, **kwargs):
        dataset_name = kwargs.pop('dataset_name', None)
        dec_type = kwargs.pop('dec_type', None)
        num_interps = kwargs.pop('num_interps', 0)
        num_hybrid_iters = kwargs.pop('num_hybrid_iters', 0)
        if dataset_name == 'synthetic_biggan' and dec_type == 'biggan':
            if num_hybrid_iters > 0:
                return HybridEvaluatorBigGan(*args, **kwargs)
            else:
                return TrainerBigGan(*args, **kwargs)
        elif dataset_name == 'synthetic_stylegan' and dec_type == 'stylegan':    
            return TrainerStyleGan(*args, **kwargs)
        elif dataset_name == 'synthetic_nvae' and dec_type == 'nvae':    
            return TrainerNVae(*args, **kwargs)
        elif dataset_name == 'text':
            if dec_type == 'biggan':
                return TextHybridEvaluatorBigGan(*args, **kwargs)
            elif dec_type == 'stylegan':
                return TextHybridEvaluatorStyleGan(*args, **kwargs)
            elif dec_type == 'nvae':
                return TextHybridEvaluatorNVae(*args, **kwargs)
            else:
                raise NotImplementedError
        elif dataset_name == 'image':
            if dec_type == 'biggan':
                if num_interps > 0:
                    return InterpImageHybridEvaluatorBigGan(*args, **kwargs)
                else:
                    return ImageHybridEvaluatorBigGan(*args, **kwargs)
            elif dec_type == 'stylegan':
                if num_interps > 0:
                    return InterpImageHybridEvaluatorStyleGan(*args, **kwargs)
                else:
                    return ImageHybridEvaluatorStyleGan(*args, **kwargs)
            else:
                raise NotImplementedError
        else:    
            raise NotImplementedError
