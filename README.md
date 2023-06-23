## Othello-RL

In this assignment, I implemented an Othello agent with the application of Deep Q-Network and Prioritized Experience Replay.

For state-value function, the agent's algorithm uses a Q function for predictions and a separate Q' function for generating targets, both of which are neural networks with the same architecture. The Q-network is trained constantly, while its weights are soft-updated to the Q'-network.

For memory buffer, Sum Tree data structure is used to store the agent's experience samples with their corresponding priorities. When the memory buffer reaches its capacity, it removed the sample with the lowest priority.

At the beginning of the training process, the agent's memory is filled up with experience samples collected from random agent vs minimax agent. Then, the agent starts improving its action policy by replaying batches of memory samples to fit the neural networks.

## Structure

- Othello.py: Othello logic game and environment.
- DeepQNetwork.py: Agent and environment interaction.
- Minimax.py: Minimax algorithm
- SumTree.py: Memory buffer's data structure.
- Game.py: Game interface.

## References

- [Reversi_visualize](https://github.com/hieugiaosu/reversi_visualize/tree/main). Thanks Hieu for allowing us to use your interface.
- [Let’s make a DQN series](https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/).
