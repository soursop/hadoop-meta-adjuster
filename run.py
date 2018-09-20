import tensorflow as tf
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import model as md
import utils as ut
from sympy import Symbol, solve
from sympy.plotting import plot



plt.interactive(False)

CSV_COLUMN_NAMES = [
    'BYTES_READ','BYTES_WRITTEN','COMBINE_INPUT_RECORDS','COMBINE_OUTPUT_RECORDS','COMMITTED_HEAP_BYTES'
    ,'CPU_MILLISECONDS','DAY_OF_MONTH','DAY_OF_WEEK','ELAPSED_MILLISECONDS','FAILED_SHUFFLE'
    ,'GC_TIME_MILLIS','HOUR_OF_DAY','LARGE_READ_OPS','MAP_INPUT_RECORDS','MAP_OUTPUT_BYTES'
    ,'MAP_OUTPUT_MATERIALIZED_BYTES','MAP_OUTPUT_RECORDS','MAP_SKIPPED_RECORDS','MERGED_MAP_OUTPUTS','PHYSICAL_MEMORY_BYTES'
    ,'READ_OPS','REDUCE_INPUT_GROUPS','REDUCE_INPUT_RECORDS','REDUCE_OUTPUT_RECORDS','REDUCE_SHUFFLE_BYTES'
    ,'REDUCE_SKIPPED_GROUPS','REDUCE_SKIPPED_RECORDS','SEGMENT_SIZE','SHUFFLED_MAPS','SPILLED_RECORDS'
    ,'SPLIT_RAW_BYTES','VIRTUAL_MEMORY_BYTES','WRITE_OPS','YEAR','TOTAL_LAUNCHED_REDUCES'
]

# EXPECT = ['POW', 'TOTAL_LAUNCHED_REDUCES']
EXPECT = ['TOTAL_LAUNCHED_REDUCES']
INPUT_COLUMNS = ['AVG_READ', 'DAY_OF_MONTH', 'HOUR_OF_DAY']
# INPUT_COLUMNS = ['AVG_READ', 'DAY_OF_MONTH', 'HOUR_OF_DAY', 'SEGMENT_SIZE']
# INPUT_COLUMNS = ['AVG_READ']
OUTPUT_COLUMN = 'ELAPSED_MILLISECONDS'
OUTPUT_COLUMNS = [OUTPUT_COLUMN]

train = pd.read_csv("mr2_part.csv", header=0)
train['AVG_READ'] = train['BYTES_READ'] / train['SEGMENT_SIZE']
test = pd.read_csv("mr1_part.csv", header=0)
test['AVG_READ'] = test['BYTES_READ'] / test['SEGMENT_SIZE']
# data['POW'] = data['TOTAL_LAUNCHED_REDUCES'].pow(2)

tf.logging.set_verbosity(tf.logging.INFO)


def extract(sess, normalizer, feature, weight, bias):
    a, bias, c = (weight[-2][0], weight[-1][0], np.sum(bias) - np.sum(weight[:-2]))
    x = Symbol('x')
    y = a * x ** 2 + bias * x

    expect, _ = solve(y, symbols={y: -c})
    # print(expect)
    plot(y,(x,-5,5))
    df = pd.DataFrame([np.append(np.full((1, len(INPUT_COLUMNS)), 0), [expect * expect, expect])], columns=INPUT_COLUMNS + EXPECT)
    return pd.DataFrame(normalizer.restore(sess, feature, df), columns=INPUT_COLUMNS + EXPECT)


log_dir = "/Users/1002707/tensor_temp"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

model = md.DNNGradient(md.Handler(train[INPUT_COLUMNS + EXPECT], train[OUTPUT_COLUMNS]))
# estimate = pd.DataFrame([[9.0272461E9 / 6410, i] for i in range(500, 2000)], columns=INPUT_COLUMNS+EXPECT)
estimate = pd.DataFrame([np.append(train.loc[300][INPUT_COLUMNS].values, [i]) for i in range(500, 1000)], columns=INPUT_COLUMNS + EXPECT)
# result = model.run(training_epochs=100, extract=extract)
result = model.run(estimate, log_dir=log_dir)

# print(result.loc[0]['TOTAL_LAUNCHED_REDUCES'])







