from multiprocessing import Process
import time

def process1():
    for i in range(5):
        time.sleep(0.5)
        print("process1:" + str(i))
        # python2.7では「print "process1:" + str(i)」でもOK

def process2():
    for i in range(5):
        time.sleep(0.7)
        print("process2:" + str(i))

def process3():
    for i in range(5):
        time.sleep(0.9)
        print("process3:" + str(i))

if __name__ == '__main__':

    process1 = Process(target=process1, args=())
    process2 = Process(target=process2, args=())
    process3 = Process(target=process3, args=())

    process1.start()
    process2.start()
    process3.start()

    process1.join()
    process2.join()
    process3.join()

    print("process ended")
