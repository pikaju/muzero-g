import pandas as pd
import matplotlib.pyplot as plt

ax = plt.gca()

frame = pd.DataFrame()
for test in range(0, 32):
    url = 'http://localhost:6006/data/plugin/scalars/scalars?tag=1.Total+reward%2F1.Total+reward&run=run{}&format=csv'.format(
        test)
    print(url)
    data = pd.read_csv(url)
    data = data.rename(columns={'Value': 'reward'})
    url = 'http://localhost:6006/data/plugin/scalars/scalars?tag=2.Workers%2F2.Training+steps&run=run{}&format=csv'.format(
        test)
    print(url)
    steps = pd.read_csv(url)
    steps = steps.rename(columns={'Value': 'training_step'})
    merged = data.merge(steps, left_on='Step', right_on='Step')

    frame = pd.concat([frame, merged])

# frame['training_step'] -= 5000
# frame = frame[frame['training_step'] >= 0]
frame = frame[frame['training_step'] <= 10000]
frame = frame[frame['training_step'] >= 9500]

# frame['reward'] *= 3.0
print(
    frame['reward'].mean(),
    frame['reward'].std(),
)
