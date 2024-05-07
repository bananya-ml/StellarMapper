import numpy as np
import torch



def generate_data(type:str='', subtype:str='', num_features:int=0, num_samples:int=0, 
                   num_targets:int=1, num_classes:int=0, split:float=0.8, contamination:float=0.0, freq_range:list=[]):
    '''
    type: str
        The type of data to be generated (default: tabular) 
    '''
    if type == 'tabular':
        assert all(x > 0 for x in [num_classes, num_features, num_samples, num_targets])
        if subtype == 'anomaly':
            assert contamination != 0.0 and contamination<1
            x = np.random.rand(num_samples, num_features)
            y = np.random.rand(num_samples, num_targets)
            y[:] = 1
            y.flat[np.random.choice(num_samples * num_targets, int((contamination * 100)/100 * num_samples), replace=False)] = -1
            
            size = int(num_samples * split)

            x_train, x_test = np.split(x, [size])
            y_train, y_test = np.split(y, [size])
            
        if subtype == 'classification':
            pass
        if subtype == 'regression':
            pass
        if subtype == 'ts':
            pass
    
    if type == 'image':
        pass
        if subtype == 'anomaly':
            pass
        if subtype == 'classification':
            pass
        if subtype == 'detection':
            pass    
    
    if type == 'spectra':
        pass
        
    return x_train, y_train, x_test, y_test
