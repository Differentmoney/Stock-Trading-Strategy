from cProfile import run
import datetime as dt
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import experiment1 as e1
import experiment2 as e2
import ManualStrategy as ms

def author():
    return 'axiao31'


# Call helper to run and generate plots for In/Out samples for manual strategy
def run():
    # in sample
    ms.ManualStrategy.evaluate(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), title = "in sample")
    # out sample
    ms.ManualStrategy.evaluate(sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31), title = "out sample")

if __name__ == "__main__":
    # Manual Strategy In/Out sample
    run()
    # Experiemnt trials
    e1.experiment1()
    e2.experiment2()