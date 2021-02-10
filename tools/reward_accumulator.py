import pandas as pd
import matplotlib.pyplot as plt

ax = plt.gca()

for run in ['lr0.0', 'lr0.5', 'lr1.0']:
    frame = pd.DataFrame()
    for test in range(0, 25):
        url = 'http://0.0.0.0:6006/data/plugin/scalars/scalars?tag=1.Total+reward%2F1.Total+reward&run={}%2Frun{}&format=csv'.format(
            run, test)
        print(url)
        data = pd.read_csv(url)
        data = data.rename(columns={'Value': 'reward'})
        url = 'http://0.0.0.0:6006/data/plugin/scalars/scalars?tag=2.Workers%2F2.Training+steps&run={}%2Frun{}&format=csv'.format(
            run, test)
        print(url)
        steps = pd.read_csv(url)
        steps = steps.rename(columns={'Value': 'training_step'})
        merged = data.merge(steps, left_on='Step', right_on='Step')

        frame = pd.concat([frame, merged])

    # frame['training_step'] -= 5000
    # frame = frame[frame['training_step'] >= 0]
    frame = frame[frame['training_step'] <= 30000]

    frame = frame.groupby(pd.cut(frame['training_step'], 32)).mean()
    frame['reward'] *= 3.0
    frame.plot(x='training_step', y='reward', ax=ax)
    frame[['training_step', 'reward']].to_csv('{}.csv'.format(run), index=False)

plt.show()
