import time
import datetime
from dateutil.relativedelta import relativedelta




class Timer(object):
    """Computes elapsed time."""
    def __init__(self, name='default'):
        self.name = name
        self.running = True
        self.total = 0
       
        self.start_time = datetime.datetime.now()
        print("<> <> <> Starting Timer [{}] at {} <> <> <>".format(self.name,self.start_time))
 

    def remains(self, total_task_num,done_task_num):
        now  = datetime.datetime.now()
        #print(now-start)  # elapsed time
        left = (total_task_num - done_task_num) * (now - self.start_time) / done_task_num
        sec = int(left.total_seconds())
        
        rt = relativedelta(seconds=sec)
     
        return "remaining time: {:02d} hours {:02d} minutes {:02d} seconds".format(int(rt.hours), int(rt.minutes), int(rt.seconds))