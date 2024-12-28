# !pip install foolbox
# Programa rodou no colab
# Nesse trabalho, a chamada da função generate_adversarial_examples foi no arquivo bci2a.py,
# na função __within_subject_experiment do trabalho: https://github.com/edw4rdyao/eeg_mi_dl
import foolbox as fb
import torch

def generate_adversarial_examples(model, test_X, test_y, eps=0.03):
    """
    Gera exemplos adversariais usando Foolbox e FGSM.

    Args:
    - model: Modelo compatível com PyTorch.
    - test_X: Dados de teste.
    - test_y: Labels de teste.
    - eps: Intensidade da perturbação adversarial.

    """
    fmodel = fb.PyTorchModel(model.module_, bounds=(test_X.min(), test_X.max()))

    device = next(model.module_.parameters()).device

    test_X_torch = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y_torch = torch.tensor(test_y, dtype=torch.int64).to(device)

    attack = fb.attacks.FGSM()
    adversarial_X, _, success = attack(fmodel, test_X_torch, test_y_torch, epsilons=eps)

    return adversarial_X.cpu().numpy()  
