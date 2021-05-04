import numpy as np
import torch 

def get_weights_transform_for_sample(num_of_classes,samples_per_class,b_labels):
    weights_for_samples=get_weights_inverse_num_of_samples(num_of_classes,samples_per_class)
    b_labels=b_labels.to('cpu').numpy()
    weights_for_samples=torch.tensor(weights_for_samples).float() 
    weights_for_samples=weights_for_samples.unsqueeze(0)
    weights_for_samples=torch.tensor(np.array(weights_for_samples.repeat(b_labels.shape[0],1)*b_labels))
    weights_for_samples=weights_for_samples.sum(1)
    weights_for_samples=weights_for_samples.unsqueeze(1)
    weights_for_samples=weights_for_samples.repeat(1,num_of_classes)
    return weights_for_samples



def get_weights_inverse_num_of_samples(num_of_classes,samples_per_class,power=1):
    weights_for_samples=1.0/np.array(np.power(samples_per_class,power))
    weights_for_samples=weights_for_samples/np.sum(weights_for_samples)*num_of_classes
    weights_for_samples=torch.tensor(weights_for_samples, dtype=torch.float32,device=torch.device("cuda"))
    return weights_for_samples