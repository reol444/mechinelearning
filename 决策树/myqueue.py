import queue

class myqueue:
    #构造函数
    def __init__(self):
        self.queue1 = queue.Queue() #队列1
        self.queue2 = queue.Queue() #队列2
        self.size=0 #队列大小

    def put(self,obj): #入队操作，模拟进栈
        if self.queue1.empty():
            self.queue2.put(obj)
            self.size += 1
        else:
            self.queue1.put(obj)
            self.size += 1

    def get(self): #出对操作，模拟出栈
        if self.size==0:
            return
        if self.queue1.empty():
            while self.queue2.qsize() > 1:
                self.queue1.put(self.queue2.get())
            self.size -= 1
            return self.queue2.get()
        else:
            while self.queue1.qsize() > 1:
                self.queue2.put(self.queue1.get())
            self.size -= 1
            return self.queue1.get()

    def empty(self): #判断队列是否为空
        if self.size==0:
            return True
        else:
            return False