
import os

class PrintIfVerbose():
    def __init__(self, verbose):
        self.verbose = verbose
    
    def __call__(self, s):
        if self.verbose:
            print(s)
