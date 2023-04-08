# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gymnasium as gym
import numpy as np

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODE = 25000
SHOW_EVERY = 2000

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("MountainCar-v0", render_mode="human")

    # observation, info = env.reset(seed=42)
    # print(env.observation_space.high)
    # print(env.observation_space.low)
    # print(env.action_space.n)
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/ DISCRETE_OS_SIZE


    def get_discreet_state(state):
        # print(type(state[0]))
        # print(state)
        # print(type(env.observation_space.low))
        # print(type(discrete_os_win_size))
        discreet_state = (state - env.observation_space.low) / discrete_os_win_size
        # print(discreet_state)
        return tuple(discreet_state.astype(np.int64))



    # print(discreet_state)
    q_table = np.random.uniform(high=0, low=-2, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
    # print(np.argmax(q_table[discreet_state]))




    for episode in range(EPISODE):
        observation, info = env.reset()
        # print(observation)
        discreet_state = get_discreet_state(observation)
        done = False
        if (episode % SHOW_EVERY == 0):
            print(episode)
            render = True
        else:
            render = False
        while not done:
            action = np.argmax(q_table[discreet_state])
            new_state, reward, termminated, truncated, info = env.step(action)
            new_discreet_state = get_discreet_state(new_state)
            done = termminated or truncated
            if render:

                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discreet_state])
                current_q = q_table[discreet_state + (action, )]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discreet_state + (action, )] = new_q

            elif new_state[0] >= env.goal_position:
                print("we made it on {} episodes".format(episode))
                q_table[discreet_state + (action,)] = 0

            discreet_state = new_discreet_state




    # print(q_table.shape)

    # print(q_table)
    # for _ in range(1000):
    #     env.render()
    #     action = env.action_space.sample()
    #
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(observation)
    #     env.render()
    #     # print(observation)
    #
    #     if terminated or truncated:
    #         observation, info = env.reset()
    # env.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
