import os
import yaml
from run_utils.logger import print_log

def recursive_key_value_assign(d,ks,v):
    """
    para:
        d {dict}
        ks {list} 
        v {value}
    """
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]], ks[1:], v)
    elif len(ks) == 1:
        d[ks[0]] = v

def load_config(checkpoint_dir, batch_size=None, max_epoch=None, data_path=None, arg_configs=[], save=False, logger=None):
    """
    parameter from config file

    parameter:
        checkpoint_dir {str}
    
    important:
        batch_size
        max_epoch
        data_path -- path to acronym
        arg_configs{list}
        save{bool} -- whether to save the train config file
    
    return:
        [dict]
    """
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    # print(config_path)
    config_path = config_path if os.path.exists(config_path) else os.path.join(os.path.dirname(os.path.dirname(__file__)),'cfgs', 'config.yaml')
    
    print_log(config_path, logger=logger)

    with open(config_path, 'r') as f:
        global_config = yaml.safe_load(f)
    
    for conf in arg_configs:
        k_str, v =conf.split(":")
        try:
            v = eval(v)
        except:
            pass
        
        ks = [int(k) if k.isdigit() else k for k in k_str.split('.')]
    
        recursive_key_value_assign(global_config, ks, v)
    
    if batch_size is not None:
        global_config['OPTIMIZER']['batch_size'] = int(batch_size)
    if max_epoch is not None:
        global_config['OPTIMIZER']['max_epoch'] = int(max_epoch)
    if data_path is not None:
        global_config['DATA']['data_path'] = data_path

    # global_config['DATA']['classes'] = None

    if save:
        with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
            yaml.dump(global_config, f)

    return global_config