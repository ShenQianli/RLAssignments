import numpy as np
from matplotlib import pyplot as plt

np.random.seed(233)


class RandomWalk(object):
    """
        A random walk environment. Please refer to README or Lecture 4 Page 19 for more details.
        Here we use 0-4 to denote state A-E and -1, 5 for the terminal states.
    """
    def __init__(self):
        self._state = 2
        self.reset()

    def reset(self):
        self._state = 2
        return self._state

    def step(self):
        reward = 0
        self._state = self._state + int(np.sign(np.random.rand() - 0.5))
        if self._state == 5:
            reward = 1
        return self._state, reward


TRUE_VALUES = np.array([1/6, 1/3, 1/2, 2/3, 5/6])


def episode(env):
    states = []
    rewards = []
    state = env.reset()
    while state in [0, 1, 2, 3, 4]:
        states.append(state)
        state, reward = env.step()
        rewards.append(reward)
    return np.array(states), np.array(rewards)


def check_env(env):
    print('Checking the environment')
    for i in range(10):
        states, rewards = episode(env)
        print('Episode {}: states: {}, rewards: {}'.format(i, states, rewards))


def rms_error(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


def mc(states_col, rewards_col, alpha=.03, plot=False):
    preds = np.ones(5) * 0.5
    rms_errors = []
    for i, states, rewards in zip(np.arange(len(states_col)), states_col, rewards_col):
        if i in [0, 1, 10, 100]:
            if plot:
                plt.plot(np.arange(5), preds, label='n={}'.format(i))
        g = rewards[-1]
        for s in states:
            preds[s] = preds[s] + alpha * (g - preds[s])
        rms_errors.append(rms_error(preds, TRUE_VALUES))
    if plot:
        plt.plot(np.arange(5), TRUE_VALUES, label='True Values')
        plt.xlabel('State')
        plt.ylabel('Estimated Value')
        plt.title(r'Random Walk Example: $MC, \alpha={}$'.format(alpha))
        plt.xticks(np.arange(5), ['A', 'B', 'C', 'D', 'E'])
        plt.legend()
        plt.tight_layout()
        plt.savefig('mc.pdf')
        plt.show()
    return np.array(rms_errors)


def td(states_col, rewards_col, alpha=.03, plot=False):
    preds = np.ones(5) * 0.5
    rms_errors = []
    for i, states, rewards in zip(np.arange(len(states_col)), states_col, rewards_col):
        if i in [0, 1, 10, 100]:
            if plot:
                plt.plot(np.arange(5), preds, label='n={}'.format(i))
        g = rewards[-1]
        for j in range(len(states) - 1):
            preds[states[j]] = preds[states[j]] + alpha * (preds[states[j+1]] - preds[states[j]])
        preds[states[-1]] = preds[states[-1]] + alpha * (g - preds[states[-1]])
        rms_errors.append(rms_error(preds, TRUE_VALUES))
    if plot:
        plt.plot(np.arange(5), TRUE_VALUES, label='True Values')
        plt.xlabel('State')
        plt.ylabel('Estimated Value')
        plt.title(r'Random Walk Example: $TD(0), \alpha={}$'.format(alpha))
        plt.xticks(np.arange(5), ['A', 'B', 'C', 'D', 'E'])
        plt.legend()
        plt.tight_layout()
        plt.savefig('td.pdf')
        plt.show()

    return rms_errors


def sweep(states_col, rewards_col):
    exps = [['mc', .01], ['mc', .02], ['mc', .03], ['mc', .04], ['td', .05], ['td', .1], ['td', .15]]
    for exp in exps:
        alg, alpha = exp
        compute_rms_errors = mc if alg == 'mc' else td
        rms_errors = compute_rms_errors(states_col, rewards_col, alpha=alpha, plot=False)
        plt.plot(np.arange(101), smooth(rms_errors, 0.8), label=r'${} \quad \alpha={}$'.format(alg, alpha))
    plt.title('MC v.s. TD')
    plt.xlabel('Episodes')
    plt.ylabel('RMS error')
    plt.xticks([0, 25, 50, 75, 100])
    plt.yticks([0, .05, .1, .15, .2, .25])
    plt.legend()
    plt.savefig('mc_vs_td.pdf')
    plt.show()


def smooth(a, weight):
    smoothed = [a[0]]
    for x in a[1:]:
        last = smoothed[-1]
        smoothed.append(weight * last + (1 - weight) * x)
    return smoothed


if __name__ == '__main__':
    env = RandomWalk()
    check_env(env)
    states_col, rewards_col = [], []
    for i in range(101):
        states, rewards = episode(env)
        states_col.append(states)
        rewards_col.append(rewards)
    mc(states_col, rewards_col, plot=True)
    td(states_col, rewards_col, plot=True)
    sweep(states_col, rewards_col)



