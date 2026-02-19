
from xml.parsers.expat import model
from preamble import *
from BackBoneModels import BaseModel, MidModel
from NeuralIsingEdgeModel import BradleyTerryEdgeModel, NeuralIsingEnergy, NeuralIsingEnergyDeclarative_J, NeuralIsingRegularizer, NeuralIsingEnergyDeclarative

MODEL_DICT = {
    'base': BaseModel,
    'mid': MidModel,
    'neural_ising_reg': NeuralIsingRegularizer,
    'neural_ising_energy':      NeuralIsingEnergy,
    'neural_ising_energy_declarative':      NeuralIsingEnergyDeclarative,
    'neural_ising_energy_declarative_J':      NeuralIsingEnergyDeclarative_J,
    "bradley_terry_edge": BradleyTerryEdgeModel,
}
def build_model(cfg):
    kwargs = {"num_classes": cfg.get('num_classes', 10)}
    for k in cfg.get('model', {}):
        if k != 'name':
            kwargs[k.replace('model.', '')] = cfg['model'][k]
    print("Building model with parameters:", kwargs)
    if 'device' not in kwargs:
        kwargs['device'] = device
    model_name = cfg.get('model.name', 'neural_ising_reg')
    if model_name not in MODEL_DICT:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from {list(MODEL_DICT.keys())}.")
    model = MODEL_DICT[model_name](**kwargs).to(device)
    if os.path.exists(os.path.join(cfg.get('save_dir', ''), 'best.pth')):
        checkpoint = torch.load(os.path.join(cfg.get('save_dir', ''), 'best.pth'), map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model checkpoint from {os.path.join(cfg.get('save_dir', ''), 'best.pth')}")
    return model