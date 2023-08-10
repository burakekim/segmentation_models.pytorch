import numpy as np
from skimage import exposure
import torch
from matplotlib import pyplot as plt

def _prepare_image(attr_visual):
    # together with the masked_image logic, from: https://github.com/pytorch/captum/blob/56abc960f2591ebe519ca2425d8acd45c99e9ef5/captum/attr/_utils/visualization.py#L100
    return np.clip(attr_visual.astype(int), 0, 255)

def stretch(im):
    p1, p99 = np.percentile(im, (1, 99))
    J = exposure.rescale_intensity(im, in_range=(p1, p99))
    J = J/J.max()
    return J

def preproc(inp):
    a = inp.squeeze()
    a = a.detach().numpy()
    a = np.transpose(a,(1,2,0))
    return stretch(a)

def wrapper(inp):
    return model(inp).sum(dim=(2,3))

def _prepare_image(attr_visual):
    # together with the masked_image logic, from: https://github.com/pytorch/captum/blob/56abc960f2591ebe519ca2425d8acd45c99e9ef5/captum/attr/_utils/visualization.py#L100
    return np.clip(attr_visual.astype(int), 0, 255)

def min_max_norm(input):
    
    mn = np.min(input)
    mx = np.max(input)
    inp_norm = (input - mn) * (1.0 / (mx - mn))
    return inp_norm 

def transpose_1_2_0(input):
    tr_inp = np.transpose(input,(1,2,0))
    return tr_inp

def image_to_model_input(input):
    tensor_unsq = torch.Tensor(input).unsqueeze(0)
    return tensor_unsq

def model_output_to_np_image(inp):
    squeezed = inp.squeeze(0)
    squeezed_np = squeezed.detach().numpy()
    squeezed_np_tr = np.transpose(squeezed_np,(1,2,0))
    return squeezed_np_tr 

def model_output_to_binary(inp):
    squeezed = inp.squeeze(0)
    squeezed_np = squeezed.detach().numpy()
    squeezed_np_tr = np.transpose(squeezed_np,(1,2,0))
    squeezed_np_tr[squeezed_np_tr>0.5] = 1
    squeezed_np_tr[squeezed_np_tr<=0.5] = 0
    return squeezed_np_tr

def expand_dim(input, dim):
    expanded = np.expand_dims(input, axis=dim)
    return expanded

def model_to_eval(mdl):
    mdl.eval()
    print("Model in eval mode")

class ForwardHook:
    """
    Helper class to define a forward hook. You can use this e.g. as follows:

    hook = ForwardHook()
    model.layer_x.register_forward_hook(hook.save_output)
    print(hook.output)
    """
    output = None

    def save_output(self, model, input_, output):

        self.output = output#.detach().cpu()
        
def wrapper(inp):
    return model(inp).sum(dim=(2,3))

def preproc(inp):
    a = inp.squeeze()
    a = np.transpose(a,(1,2,0))
    return stretch(a)


def agg_segmentation_wrapper(inp, model):
    model_out, clsf_out = model(inp)
    out_max = torch.argmax(model_out, dim=1, keepdim=True)

    # Creates binary matrix with 1 for original argmax class for each pixel
    # and 0 otherwise. Note that this may change when the input is ablated
    # so we use the original argmax predicted above, out_max.
    selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
    return (model_out * selected_inds).sum(dim=(2,3))
   
    
def viz_ims(*images, titles, n_of_images, fig_size, export_file_id):
    list_images = list(images)
    #n_of_images = len(images)
    #print(n_of_images)
    list_titles = list(titles)

    fig, ax = plt.subplots(1, ncols = n_of_images, figsize = fig_size)
    
    for i in range(n_of_images):
        ax[i].imshow(list_images[i])
        ax[i].axis(False)
        ax[i].set_title(list_titles[i])
        
    if export_file_id is not None:
        plt.savefig(r'C:\Users\burak\Desktop\logits' + '\{}'.format(export_file_id), dpi=100)   
        plt.tight_layout()
        plt.show()        

def viz_ims_w_prob_osm(images, titles, n_of_images, fig_size, export):
    file_id = np.random.randint(1000)
    
    COLOR = 'white'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    plt.rcParams['figure.facecolor'] = 'black'
    
    list_images = list(images)
    list_titles = list(titles)

    fig, ax = plt.subplots(1, ncols = n_of_images, figsize = fig_size, constrained_layout=True)
    
    for idx, i in enumerate(range(len(images))):
        ax[idx].imshow(images[i])
        ax[idx].axis(False)
        ax[idx].set_title(list_titles[i])
        
    if export is not False:
        plt.savefig(r'C:\Users\burak\Desktop\OSM' + '\{}'.format(file_id), dpi=100)   
        plt.tight_layout()
        plt.show()
        
    if export is False:
        plt.tight_layout()
        plt.show()
        
    return fig,ax

def get_band_and_name(i):
    name = "band" +"_{}".format(i)
    band = model_out[:,:,i]
    return name,band

def process_to_viz_14_band(image,i,apply_stretch): # input shape of images[i] torch.Size([1, 14, 256, 256])
    # TODO:  change 14 to "n"
    if apply_stretch == True:
        out = stretch(image[i][:,2:5,:,:].detach().numpy().transpose(0,2,3,1)) #output shape (1, 256, 256, 3)
    elif apply_stretch == False:
        out = image[i][:,2:5,:,:].detach().numpy().transpose(0,2,3,1) #output shape (1, 256, 256, 3)
    return out

def serialize_viz_14_band(images,apply_stretch):
    """
    aa
    """
    lissst = []
    n_of_images = len(images)
    for i in range(n_of_images):
        lissst.append(process_to_viz_14_band(images,i,apply_stretch))
    concated = np.concatenate(lissst,0)
    return concated #shape (16, 256, 256, 3)


def index_modalities(bands, modality_from_dict):
    occurance_index = list(map(lambda x : 1 if x in modality_from_dict else 0, bands))    
    max_indexes = [i for i, x in enumerate(occurance_index) if x == max(occurance_index)]
    return max_indexes