from torch.nn import Module
import torch.nn.functional as F
from torchvision.transforms import transforms
from .interpreter import Interpreter
from .interpreter import DecisionInterpreter
import torch.nn as nn
import torch

class OcclusionSensitivity(Interpreter):
    """
    OcclusionSensitivity is a decision-based intepretability method which obstructs
    parts of the input in order to see what influence these regions have to the
    output of the model under test.
    """
    def __init__(self, model: Module, classes, preprocess: transforms.Compose, input_size, block_size, fill_value, target, bands, occlusion_mode, batch_size, occlusion_modality):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param input_size: The expected 2D input size by the given model (e.g. (256, 256))
        :param block_size: The size of the 2D block which acts as occlusion.
                            This should be a divisor of each dimension of the input size! (e.g. (7, 7))
        :param fill_value: The value for the occlusion block
        :param target: The target class of the expected input
        """
        DecisionInterpreter.__init__(self, model, classes, preprocess)
        self.input_size = input_size
        self.block_size = block_size
        self.fill_value = fill_value
        self.target = target
        self.bands_selected = bands
        self.occlusion_mode = occlusion_mode
        #self.encode_input = encode_input
        self.batch_size = batch_size
        self.occlusion_modality = occlusion_modality

        self.learner = self.model #.eval() ### put in the eval mode

        if self.input_size[0] % self.block_size[0] != 0 or self.input_size[1] % self.block_size[1] != 0:
            raise ValueError("The block size should be a divisor of the input size.")

    def _generate_occlused_input(self, x):
        out = []

        rows = int(self.input_size[0] / self.block_size[0])
        columns = int(self.input_size[1] / self.block_size[1])
        
        
        if self.occlusion_mode == 'spatial':
            for row in range(rows):
                for column in range(columns):
                    new_x = x.clone()
                    new_x[0][:, row *  self.block_size[0]: (row + 1) *  self.block_size[0], column *  self.block_size[1]: (column + 1) *  self.block_size[1]] =  self.fill_value
                    out.append(new_x)
        
        elif self.occlusion_mode == 'spectral':
            new_x = x.clone()
            if len(self.bands_selected) == 1:
                new_x[0][self.bands_selected, :, :] =  self.fill_value
            elif len(self.bands_selected) != 1:
                new_x[0][self.bands_selected[0]:self.bands_selected[1], :, :] = self.fill_value
            out.append(new_x)

        else:
            raise NotImplemented("Please pick either 'spatial' or 'spectral'.")
                
        return out

    def _prepare_occlusionencoder(self):
        occlusion_encoder = occlusion_encode()
        return occlusion_encoder  

    def _compute_probabilities(self, x):
        probabilities = []
        inp = x[0] #.unsqueeze(0) # inp = 4,14,256,256
        #print("inp.shape", inp.shape)
        out = self.learner.forward(inp, torch.zeros(self.batch_size,self.occlusion_modality))
        logits, clsf = out
        #print("clsf", clsf)
        #print("clsf", clsf.shape)
        rounded_scores = clsf#[self.target] ### # for only one neuron delete both indexings
        #print("rounded_scores",rounded_scores.shape)

        #print("rounded score",rounded_scores)
        probabilities.append(rounded_scores)
        return probabilities#.squeeze()

    def interpret(self, x):
        #x = self._execute_preprocess(x)

        occluded_input = self._generate_occlused_input(x)
        probabilities = self._compute_probabilities(occluded_input)

        return probabilities# occluded_input