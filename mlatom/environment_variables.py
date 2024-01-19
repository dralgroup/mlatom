import os 
from multiprocessing import cpu_count
# rename and  rewrite

class environment_variables():
    def __init__(self,**kwargs):
        # Check CPU type
        self.check_cpu()
        # By default use all the CPU cores
        # self.set_nthreads(cpu_count())
        self.nthreads = cpu_count()

    def set_nthreads(self, nthreads):
        self.nthreads = nthreads
        os.environ["OMP_NUM_THREADS"] = str(self.nthreads)
        os.environ["MKL_NUM_THREADS"] = str(self.nthreads)
        os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = str(self.nthreads)

    def get_nthreads(self):
        return self.nthreads
    
    def check_cpu(self):
        try:
            f_cpu = open('/proc/cpuinfo')
            for line in f_cpu:
                if 'AMD'.lower() in line.lower():
                    os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
                    break
            f_cpu.close()
        except: pass
