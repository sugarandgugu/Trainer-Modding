from transformers import TrainerCallback,PrinterCallback,TrainingArguments,TrainerState,TrainerControl
import time
import datetime
'''
    PrinterCallback 或 ProgressCallback，用于显示进度和打印日志（如果通过TrainingArguments停用tqdm，
    则使用第一个函数；否则使用第二个）。
'''

class ProgressCallback(TrainerCallback):

    def setup(self, total_epochs, print_every=1): 
        # 通过总的epoch 设置一些日志信息
        self.total_epochs = total_epochs 
        self.current_epoch = 0
        self.epoch_start_time = None
        self.current_step = 1
        self.global_start_time = time.time()
        self.print_every=print_every
        return self

    def on_step_begin(self, args, state, control, **kwargs):
        
        avg_time_per_step = (time.time() - self.global_start_time)/max(state.global_step,1 )
        eta = avg_time_per_step * (state.max_steps-state.global_step) / 3600
        if self.current_step % self.print_every == 0:
            print(
                'epoch: ', 
                self.current_epoch,
                ', step ',
                self.current_step, 
                '/', 
                state.max_steps // self.total_epochs, 
                '||', 
                datetime.datetime.now(),
                # 记录运行时间
                '|| running time: ',
                round((time.time() - self.global_start_time)/3600,2),
                '|| ETA(hrs): ',
                round(eta,2)
                )
        self.current_step += 1

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return super().on_log(args, state, control, **kwargs)

    '''
        在每个epoch开始前调用
    '''
    def on_epoch_begin(self, args, state, control, **kwargs):
        print('[ProgressCallback]: current epoch: ', self.current_epoch,' / ', self.total_epochs)
        self.current_epoch += 1
        self.current_step = 1
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        print('[ProgressCallback]: epoch', self.current_epoch,' / ', self.total_epochs, ' done')
        print("--- %s hours ---" % ((time.time() - self.epoch_start_time)/3600))
