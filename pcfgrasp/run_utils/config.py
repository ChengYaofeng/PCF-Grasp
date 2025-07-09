import os
import yaml
from run_utils.logger import print_log

def recursive_key_value_assign(d,ks,v):
    """
    键值匹配,让最后一行的数代表的字典的值等于v
    参数：
        d {dict} --dict
        ks {list} -- 分层键的列表
        v {value} --值
    """
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]], ks[1:], v)
    elif len(ks) == 1:
        d[ks[0]] = v

def load_config(checkpoint_dir, batch_size=None, max_epoch=None, data_path=None, arg_configs=[], save=False, logger=None):
    """
    加载yaml文件夹中的所用关于训练的数据,同时覆盖arg定义的一些指定的参数

    参数：
        checkpoint_dir {str} --checkpoint路径
    
    关键参数：
        batch_size
        max_epoch
        data_path --场景文件路径
        arg_configs{list}--重写状态参数
        save{bool}--是否保存重写后的状态文件
    
    返回：
        [dict] --状态
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