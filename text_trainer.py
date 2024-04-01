from transformers import Trainer
import torch 
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from loguru import logger
from transformers.trainer_utils import EvalLoopOutput
import numpy as np
import datetime


'''
    继承Trainer类，实现自定义的一些函数，达到魔改的效果,这里只写了几个常用的方法，可以参考下面的链接，进行修改:
    https://github.com/huggingface/transformers/blob/v4.39.2/src/transformers/trainer.py#L3199
'''
class MyTrainer(Trainer):

    '''
        计算损失函数
        args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
        已check compute_loss 没问题
    '''
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # print(labels)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0], device=model.device))
        # 2分类 直接调整为2
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


    '''
        写完这个evaluation_loop发现我做的事情 在外面的compute_metrics实现更加简单,不过为了熟悉Trainer还是自己写了一个
        注意返回值一定是EvalLoopOutput类型
        args:  
            dataloader: 测试集
            description: 描述
            prediction_loss_only: 如果是True，则只返回loss
            ignore_keys: 忽略的key
            metric_key_prefix: 指标前缀
    '''
    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        ):
            """
            Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
            Works both with or without labels.
            """
            args = self.args
            model = self._wrap_model(self.model, training=False)
            self.callback_handler.eval_dataloader = dataloader
            model.eval()
            batch_size = args.per_device_eval_batch_size
            num_examples = self.num_examples(dataloader)
            print(f"***** Running evaluation loop *****")
            print(f"  Num examples = {num_examples}")
            print(f"  Batch size = {batch_size}")
            

            loss_list = []
            labels_list = []
            preds_list = []
            # evaldataset的全部样本算一次准确率 一个batch算  然后叠加
            for step, inputs in enumerate(dataloader):

                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys)

                # 一个batch算
                logits = nn.Softmax(dim=1)(logits)
                pred_labels = torch.argmax(logits, dim=1)

                loss = loss.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                pred_labels = pred_labels.detach().cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(pred_labels)


                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            metrics = {}

            test_loss = np.mean(loss_list)
            test_accuracy = accuracy_score(labels_list, preds_list)
            test_precision = precision_score(labels_list, preds_list, average='macro')
            test_recall = recall_score(labels_list, preds_list, average='macro')
            test_f1_score = f1_score(labels_list, preds_list, average='macro')

            loss_dict = {'loss': test_loss,'accuracy': test_accuracy, 'recall':test_recall, 'f1_score':test_f1_score, 'precision':test_precision}

            logger.info("==========test_loss{}==========".format(test_loss))
            logger.info("==========test_accuracy{}==========".format(test_accuracy))
            logger.info("==========test_recall{}==========".format(test_recall))
            logger.info("==========test_f1_score{}==========".format(test_f1_score))
            logger.info("==========test_precision{}==========".format(test_precision))
            
            '''
                下面的这个作用好像是在进度条返回这个metric，就是metric都会在tdqm bar上面显示
            '''
            for key, value in loss_dict.items():
                metrics['eval_' + key] = torch.tensor(value).item()

            print('[MyTrainer]: Evaluation done)')

            output = EvalLoopOutput(predictions=preds_list, label_ids=labels_list, metrics=metrics, num_samples=num_examples)
            return output
    
    '''
        这里和evaluation_loop差不多,这里没有做这个实现，只是说明，我们可以继承这个方法进行改写。
        args:
            dataloader: 测试集
            description: 描述
            prediction_loss_only: 如果是True，则只返回loss
            ignore_keys: 忽略的key
            metric_key_prefix: 指标前缀
    '''
    def prediction_loop(
            self,
            dataloader,
            description: str,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        ) -> EvalLoopOutput:
                super().prediction_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix)


    '''
        这个方法我们也可以进行改写，当模型是我们自定义的时候
        args:
            output_dir: 输出目录
            state_dict: 状态字典
    '''
    def _save(self,output_dir,state_dict):
        super()._save(output_dir,state_dict)
    
