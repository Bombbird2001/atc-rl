from mcts import mcts
from tc2_env import make_env, MCTSState


def run():
    mcts_search = mcts(timeLimit=3333)
    tc2_env = make_env(env_id=0, processes=[], auto_init_sim=False)
    current_state = MCTSState.getRootState(tc2_env, tc2_env.reset()[0])
    while True:
        actions = []
        for i in range(3):
            partial_action = mcts_search.search(current_state)
            actions.append(partial_action)
            current_state = current_state.takeAction(current_state)
        # TODO Set sim state to current state and take action

        new_state = tc2_env.step(tuple(actions))[0]
        current_state = MCTSState.getRootState(tc2_env, new_state)


if __name__ == '__main__':
    run()