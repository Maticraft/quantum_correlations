import os
from functools import wraps

from commons.data.savers import DATASETS_DIR_NAME, DICTIONARY_NAME, NEGATIVITY_BIPART_DICT_NAME, DISCORD_BIPART_DICT_NAME, NUMERICAL_SEPARABILITY_BIPART_DICT_NAME, NUM_NEAR_ZERO_EIGVALS_BIPART_DICT_NAME

def generate_with_assertion(batch_size=None):
    def _decorator(generation_function, batch_size=batch_size):
        @wraps(generation_function)
        def _wrapper(self, *args, batch_size=batch_size, **kwargs):
            try:
                save_data_dir = args[2]
            except:
                save_data_dir = kwargs['save_data_dir']
            if save_data_dir == None or kwargs.get('return_matrices', False):
                return generation_function(self, *args, **kwargs)
            
            try:
                qubits_num = args[0]
            except:
                qubits_num = kwargs['qubits_num']
            try:
                desired_examples = args[1]
                if batch_size is not None and desired_examples > batch_size:
                    args[1] = batch_size
                    if 'base_size' in kwargs:
                        kwargs['base_size'] = batch_size
            except:
                desired_examples = kwargs['examples']
                if batch_size is not None and desired_examples > batch_size:
                    kwargs['examples'] = batch_size
                    if 'base_size' in kwargs:
                        kwargs['base_size'] = batch_size

            if batch_size is None or desired_examples < batch_size:
                batch_size = desired_examples

            filepath = f"./{DATASETS_DIR_NAME}/{qubits_num}qbits/{save_data_dir}/"
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
            fps = [filepath + DICTIONARY_NAME]
            if kwargs.get('encoded', False):
                fps.append(filepath + NEGATIVITY_BIPART_DICT_NAME)
                fps.append(filepath + NUMERICAL_SEPARABILITY_BIPART_DICT_NAME)
                fps.append(filepath + NUM_NEAR_ZERO_EIGVALS_BIPART_DICT_NAME)
            if kwargs.get('discord', False):
                fps.append(filepath + DISCORD_BIPART_DICT_NAME)
        
            try:
                with open(fps[0], 'r') as f:
                    start_num_examples = len(f.readlines())
            except:
                start_num_examples = 0

            generated_examples = 0
            start_indx = kwargs.get('start_index', 0)
            kwargs['start_index'] = start_indx

            while generated_examples < desired_examples:
                generation_function(self, *args, **kwargs)
                for fp in fps:
                    try:
                        with open(fp, 'r+') as dictionary:
                            data = dictionary.readlines()
                            generated_examples = len(data) - start_num_examples
                            if generated_examples > desired_examples:
                                extra_lines = generated_examples - desired_examples
                                dictionary.seek(0)
                                dictionary.truncate()
                                dictionary.writelines(data[:-extra_lines])
                                generated_examples -= extra_lines
                    except:
                        pass
                
                kwargs['start_index'] += batch_size
            return kwargs['start_index']

        return _wrapper
    return _decorator
