"""
tester模块实现了 fastNLP 所需的Tester类，能在提供数据、模型以及metric的情况下进行性能测试。

.. code-block::

    import numpy as np
    import torch
    from torch import nn
    from fastNLP import Tester
    from fastNLP import DataSet
    from fastNLP import AccuracyMetric

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)
        def forward(self, a):
            return {'pred': self.fc(a.unsqueeze(1)).squeeze(1)}

    model = Model()

    dataset = DataSet({'a': np.arange(10, dtype=float), 'b':np.arange(10, dtype=float)*2})

    dataset.set_input('a')
    dataset.set_target('b')

    tester = Tester(dataset, model, metrics=AccuracyMetric())
    eval_results = tester.test()

这里Metric的映射规律是和 :class:`fastNLP.Trainer` 中一致的，具体使用请参考 :mod:`trainer 模块<fastNLP.core.trainer>` 的1.3部分。
Tester在验证进行之前会调用model.eval()提示当前进入了evaluation阶段，即会关闭nn.Dropout()等，在验证结束之后会调用model.train()恢复到训练状态。


"""
import time
import numpy as np
import torch
import torch.nn as nn
import pdb
import pickle

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm

from .batch import BatchIter, DataSetIter
from .dataset import DataSet
from .metrics import _prepare_metrics
from .sampler import SequentialSampler
from .utils import _CheckError
from .utils import _build_args
from .utils import _check_loss_evaluate
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from .utils import _get_model_device
from .utils import _move_model_to_device
from ._parallel_utils import _data_parallel_wrapper
from ._parallel_utils import _model_contains_inner_module
from functools import partial
from ._logger import logger

__all__ = [
    "Tester"
]

def list_intersection(la,lb):
    a = la.copy()
    b = lb.copy()
    inter = []
    while(len(a)*len(b)):
        x = a[0]
        if x in b:
            inter.append(x)
            a.remove(x)
            b.remove(x)
        else:
            a.remove(x)
    return inter

class Tester(object):
    """
    Tester是在提供数据，模型以及metric的情况下进行性能测试的类。需要传入模型，数据以及metric进行验证。
    """
    
    def __init__(self, data, model, metrics, batch_size=16, num_workers=0, device=None, verbose=1, use_tqdm=True, status='train'):
        """
        
        :param ~fastNLP.DataSet data: 需要测试的数据集
        :param torch.nn.module model: 使用的模型
        :param ~fastNLP.core.metrics.MetricBase,List[~fastNLP.core.metrics.MetricBase] metrics: 测试时使用的metrics
        :param int batch_size: evaluation时使用的batch_size有多大。
        :param str,int,torch.device,list(int) device: 将模型load到哪个设备。默认为None，即Trainer不对模型
            的计算位置进行管理。支持以下的输入:
    
            1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中,可见的第一个GPU中,可见的第二个GPU中;
    
            2. torch.device：将模型装载到torch.device上。
    
            3. int: 将使用device_id为该值的gpu进行训练
    
            4. list(int)：如果多于1个device，将使用torch.nn.DataParallel包裹model, 并使用传入的device。
    
            5. None. 为None则不对模型进行任何处理，如果传入的model为torch.nn.DataParallel该值必须为None。
    
            如果模型是通过predict()进行预测的话，那么将不能使用多卡(DataParallel)进行验证，只会使用第一张卡上的模型。
        :param int verbose: 如果为0不输出任何信息; 如果为1，打印出验证结果。
        :param bool use_tqdm: 是否使用tqdm来显示测试进度; 如果为False，则不会显示任何内容。
        """
        super(Tester, self).__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")
        
        self.metrics = _prepare_metrics(metrics)
        
        self.data = data
        self._model = _move_model_to_device(model, device=device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.logger = logger
        self.status = status
        self.g = open('../test_predictions_flat_lattice.txt','w')
        self.h = open('../test_logits_FL.pkl','wb')

        if isinstance(data, DataSet):
            self.data_iterator = DataSetIter(
                dataset=data, batch_size=batch_size, num_workers=num_workers, sampler=SequentialSampler())
        elif isinstance(data, BatchIter):
            self.data_iterator = data
        else:
            raise TypeError("data type {} not support".format(type(data)))

        # check predict
        if (hasattr(self._model, 'predict') and callable(self._model.predict)) or \
                (_model_contains_inner_module(self._model) and hasattr(self._model.module, 'predict') and
                 callable(self._model.module.predict)):
            if isinstance(self._model, nn.DataParallel):
                self._predict_func_wrapper = partial(_data_parallel_wrapper('predict',
                                                                    self._model.device_ids,
                                                                    self._model.output_device),
                                                     network=self._model.module)
                self._predict_func = self._model.module.predict  # 用于匹配参数
                print("@@ case 1\n")
            elif isinstance(self._model, nn.parallel.DistributedDataParallel):
                self._predict_func = self._model.module.predict
                self._predict_func_wrapper = self._model.module.predict  # 用于调用
                print("@@ case 2\n")
            else:
                self._predict_func = self._model.predict
                self._predict_func_wrapper = self._model.predict
                print("@@ case 3\n")
        else:
            if _model_contains_inner_module(model):
                self._predict_func_wrapper = self._model.forward
                self._predict_func = self._model.module.forward
                print("@@ case 4\n")
            else:
                self._predict_func = self._model.forward
                self._predict_func_wrapper = self._model.forward
                print("@@ case 5\n")
    
    def test(self):
        r"""开始进行验证，并返回验证结果。

        :return Dict[Dict]: dict的二层嵌套结构，dict的第一层是metric的名称; 第二层是这个metric的指标。一个AccuracyMetric的例子为{'AccuracyMetric': {'acc': 1.0}}。
        """
        # turn on the testing mode; clean up the history
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        id2char_fl = {0: '<pad>', 1: '<unk>', 2: 'a', 3: '_', 4: 'A', 5: 't', 6: 'm', 7: 'r', 8: 'v', 9: 'i', 10: 'n', 11: 'y', 12: 's', 13: 'e', 14: 'u', 15: 'p', 16: 'k', 17: 'd', 18: 'H', 19: 'c', 20: 'a_', 21: 'z', 22: 'S', 23: 'h', 24: 'R', 25: 'B', 26: 'g', 27: 'j', 28: 'D', 29: 'l', 30: 'aH', 31: 'I', 32: 'f', 33: 'T', 34: 'o', 35: 'a_a', 36: 'H_', 37: 'U', 38: 'b', 39: 'E', 40: 't_', 41: 'i_', 42: 'w', 43: 'O', 44: 'Y', 45: 'AH', 46: 'm_', 47: 'M', 48: 'q', 49: 'a_e', 50: 'a_u', 51: 'K', 52: 'u_', 53: 'N', 54: 'G', 55: 'C', 56: 'W', 57: 'A_a', 58: 'A_', 59: 'a_i', 60: 'P', 61: 'I_', 62: 'A_e', 63: 'n_', 64: 'A_u', 65: 'i_i', 66: 'H_a', 67: 'A_i', 68: 'ya_', 69: 'va_', 70: '_f', 71: 'ra_', 72: 'Q', 73: 'ma_', 74: 'aH_', 75: 'na_', 76: 'ca_', 77: 'k_', 78: 'ta_', 79: 'a_I', 80: 'A_A', 81: 'f_', 82: 'Ra_', 83: 'la_', 84: '_a', 85: 'u_u', 86: 'w_', 87: 'a_o', 88: 'ka_', 89: 'At_', 90: 'pa_', 91: 'at_', 92: 'ha_', 93: 'Da_', 94: 'ga_', 95: 'U_', 96: 'a_U', 97: 'da_', 98: 'TA_', 99: 'Ta_', 100: 'sa_', 101: 'd_', 102: 'za_', 103: 'I_i', 104: 'Sa_', 105: 'tA_', 106: 'm_m', 107: 'Ka_', 108: 'wa_', 109: 'ja_', 110: 'J', 111: 'yA_', 112: 'am_', 113: 'am', 114: 'A_I', 115: 'Ba_', 116: 'vA_', 117: 'et_', 118: 'DA_', 119: 'Ga_', 120: 'nA_', 121: 'Pa_', 122: 'ni_', 123: 'dA_', 124: 'Wa_', 125: 'qa_', 126: 'A_U', 127: 'jA_', 128: 'uByam', 129: 'lA_', 130: 'rA_', 131: 'ri_', 132: 'ti_', 133: 'aH_u', 134: 'I_I', 135: 'ava', 136: 'e_', 137: 'sA_', 138: 'pi_', 139: 'e_i', 140: 'mA_', 141: 'Am_a', 142: 'wA_', 143: 'Qa_', 144: 'ama', 145: '_h', 146: 'n_an', 147: 'gi_', 148: 'BA_', 149: 'Ca_', 150: 'Ya_', 151: 'an', 152: 'kA_', 153: 'A_o', 154: '\\xf1', 155: 'aH_a', 156: 'AH_A', 157: 'It_', 158: 'O_', 159: 'KA_', 160: 'rI_', 161: 'gA_', 162: 'taH_', 163: 'it_', 164: 'ika_', 165: 'QA_', 166: 'ft_', 167: 'AH_', 168: 'zu', 169: 'tu_', 170: 'aH_i', 171: 'O_a', 172: 'qA_', 173: 'RA_', 174: 'CA_', 175: 'aN', 176: 'At_a', 177: 'An_a', 178: '_am', 179: 'smAkam', 180: 'sya', 181: 'At_A', 182: 'di_', 183: 'am_a', 184: 'uzva', 185: 'WA_', 186: 'ayA_', 187: 'S_', 188: 'taH', 189: 'ru_', 190: 'ba_', 191: 'm_Am', 192: 'na_n', 193: 'om', 194: 'nI_', 195: 'SA_', 196: 'yAy', 197: 'ya_a', 198: 'tAy', 199: 'm_am', 200: 'asva', 201: 'Ami', 202: 'e_I', 203: 'tAt_', 204: 'At', 205: 'ni', 206: 'am_i', 207: 'ipAsi', 208: 'dI_', 209: 'wi_', 210: 'ka_k', 211: 'n_An', 212: 'Am', 213: 'Ani_', 214: 'An_', 215: 'YA_', 216: 'zA_', 217: 'om_', 218: 'hu_', 219: 'cA_', 220: 'tO_', 221: 'hi_a', 222: 'li_', 223: 'su', 224: 'Ri_', 225: 'I_a', 226: 'yAm_', 227: 'iH_', 228: 'DeBy', 229: 'sya_', 230: 'DaH_', 231: 'nam_', 232: 'Ut_', 233: 'tay', 234: 'ena_a', 235: 'mi_', 236: 'D_', 237: 'hi_', 238: 'zu_', 239: 'u_izu_', 240: '_ISam', 241: 'p_', 242: 'tAH_', 243: 'sy', 244: 'yA', 245: 'j_', 246: 'vI_', 247: 'danI', 248: 'asya', 249: 'Di_', 250: 'A_nA', 251: 'ami', 252: 'ezu', 253: 'at_a', 254: 'Am_', 255: 'ti', 256: 'yu_'}
        char2id_fl = {}
        for i in id2char_fl:
            char2id_fl[id2char_fl[i]] = i  
        eval_results = {}
        try:
            #pdb.set_trace()
            with torch.no_grad():
                if not self.use_tqdm:
                    from .utils import _pseudo_tqdm as inner_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")

                    start_time = time.time()
                    if(self.status == 'test'):
                        total = 0
                        correct = 0
                        precisions = []
                        recalls = []
                        all_logits = []
                        softmax = nn.Softmax(dim=0)
                        gold_file = open('gold_standard.txt','w')
                        pred_file = open('pred_standard.txt','w')

                    attf = open('test_attention_weights.csv','w')
                    attf.close()

                    count = 0
                    for batch_x, batch_y in data_iterator:

                        count += 1
                        if count == 3 : print(batch_x)
                
                        temp0 = []
                        temp1 = []
                        temp2 = []
                        temp2_ex = []
                        temp_inp = []
                        temp3 = batch_y['target'].tolist()[0]
                        # print("len temp3 : ",len(temp3))
                        tempx = batch_x['lattice'].tolist()[0]
                        poss = batch_x['pos_s'].tolist()[0]
                        pose = batch_x['pos_e'].tolist()[0]
                        # print('tempx : ',tempx)
                        temp5 = batch_x['target'].tolist()[0]
                        for i in range(len(temp3)):
                            # print(i)
                            temp0.append(self._model.vocabs['lattice'].to_word(tempx[i]))
                            temp1.append(self._model.vocabs['label'].to_word(temp3[i]))
                            temp_inp.append(self._model.vocabs['label'].to_word(temp5[i]))
                        
                        # print("input : ",temp_inp)
                        temp_lat = []
                        for i in range(len(tempx)):
                            temp_lat.append((self._model.vocabs['lattice'].to_word(tempx[i]),str(poss[i]),str(pose[i])))
                        # print('temp_lat : ',temp_lat)
                        temp_lat1 = [(str(t[0]), t[1], t[2]) for t in temp_lat]
                        # print('gold : ',temp1)
                        gold_sent = ''.join(temp1)
                        # gold_lattice = [lt for lt in temp_lat if lt in gold_sent]    
                        gold_lattice = gold_sent.split('_')                    
                        attf = open('test_attention_weights.csv','a')
                        _temp_lat = [';'.join(t) for t in temp_lat1]

                        attf.write(','.join(_temp_lat)+'\n')
                        attf.write(','.join(gold_lattice)+'\n')
                        attf.close()
                        

                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                        #print("batch x : ", type(batch_x))
                        #print(batch_x)
                        pred_dict = self._data_forward(self._predict_func, batch_x)
                        #print("pred dict : ", pred_dict)
                        if(self.status=='test'):
                            last_tags = pred_dict['pred'].tolist()[0]
                            seq_len = len(last_tags)
                            logits_mix = pred_dict['logits'] 
                            logits_mix = [lij.cpu() for lij in logits_mix]
                            logits_i = []
                            for j in range(seq_len-1):
                                lg = logits_mix[j]
                                lt = last_tags[j+1]
                                logits_i.append(lg.transpose(1,2)[0][lt].reshape(1,-1).numpy())
                                
                            logits_i.append(logits_mix[-1])
                            all_logits += [[softmax(torch.tensor(l)).numpy() for l in logits_i]]
                            total+=1
                            wrong_tar = []
                            wrong_pred = []
                            if(len(batch_y['target'].tolist()) != len(pred_dict['pred'].tolist())):
                                print('NOT EQUAL')
                                exit()

                            else:
                                temp0 = []
                                temp1 = []
                                temp2 = []
                                temp2_ex = []
                                temp_inp = []
                                temp3 = batch_y['target'].tolist()[0]
                                tempx = batch_x['lattice'].tolist()[0]
                                temp4 = pred_dict['pred'].tolist()[0]
                                temp5 = batch_x['lattice'].tolist()[0]
                                for i in range(len(temp3)):
                                    temp0.append(self._model.vocabs['lattice'].to_word(temp5[i]))
                                    temp1.append(self._model.vocabs['label'].to_word(temp3[i]))
                                    temp2.append(self._model.vocabs['label'].to_word(temp4[i]))
                                    temp_inp.append(self._model.vocabs['lattice'].to_word(tempx[i]))

                                s = ''
                                for j in range(len(temp2)): 
                                    c = temp2[j]
                                    if c=='_' and temp_inp[j]=='_':
                                        s += '$'
                                    else:
                                        s += c

                                temp_target = ''.join(temp1).split('_')
                                temp_pred = ''.join(temp2).split('_')
                                self.g.write(s+'\n')
                                wrong_tar=temp_target
                                wrong_pred=temp_pred
                                intersection = list_intersection(temp_target,temp_pred)
                                precisions.append(len(intersection)*1.0/len(temp_pred))
                                recalls.append(len(intersection)*1.0/len(temp_target))
                                #intersection = set(temp_target).intersection(temp_pred)
                                #precisions.append(len(intersection)*1.0/len(set(temp_pred)))
                                #recalls.append(len(intersection)*1.0/len(set(temp_target)))
                                for temp1_i in range(len(temp1)):
                                    if(temp0[temp1_i]=='_'):
                                        temp0[temp1_i]='$'
                                        if(temp2[temp1_i]=='_'):
                                            temp2[temp1_i]='$'
                                        else:
                                            temp2[temp1_i]=temp2[temp1_i].replace('_','$')

                                gold_file.write(''.join(temp0)+'\n')
                                pred_file.write(''.join(temp2)+'\n')

                            if(torch.equal(pred_dict['pred'],batch_y['target'])):
                                correct+=1
                        
                        if not isinstance(pred_dict, dict):
                            raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                            f"must be `dict`, got {type(pred_dict)}.")
                        for metric in self.metrics:
                            #print('pred_dict: ',pred_dict)
                            #print('batch_y: ',batch_y)
                            metric(pred_dict, batch_y)

                        if self.use_tqdm:
                            pbar.update()                 
                    
                    if(self.status=='test'):
                        pickle.dump(all_logits,self.h)

                    if(self.status=='test'):
                        avg_prec = np.mean(precisions)*100.0
                        avg_recall = np.mean(recalls)*100.0
                        f1_score = 2*avg_prec*avg_recall/(avg_prec + avg_recall)
                        print('Accuracy: '+str((correct/total)*100))
                        print('Precision: '+str(avg_prec))
                        print('Recall: '+str(avg_recall))
                        print('F1_score: '+str(f1_score))
                        gold_file.close()
                        pred_file.close()
                        exit()

                    for metric in self.metrics:
                        eval_result = metric.get_metric()
                        if not isinstance(eval_result, dict):
                            raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                            f"`dict`, got {type(eval_result)}")
                        metric_name = metric.get_metric_name()
                        eval_results[metric_name] = eval_result
                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
        self._mode(network, is_test=False)
        return eval_results
    
    def _mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()
    
    def _data_forward(self, func, x):
        """A forward pass of the model. """
        x = _build_args(func, **x)
        y = self._predict_func_wrapper(**x)
        return y
    
    def _format_eval_results(self, results):
        """Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + ': '
            _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
            _str += '\n'
        return _str[:-1]
