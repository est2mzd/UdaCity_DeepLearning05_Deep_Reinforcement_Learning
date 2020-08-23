import numpy as np
import random
from collections import namedtuple, deque

from model_Solution import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Check GPU Existence")
print(device)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network : メモリをGPU上に確保している
        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        # 最適化器 = Adam : 更新対象は、GPU上の重み、 qnetwork_local
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        # 分かりにくいが、t_step を UPDATE_EVERY(=4)で割った"余り"　で上書きしている
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # UPDATE_EVERY(=4)の倍数の時のみ下記を処理する
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # BATCH_SIZE(=64)個以上のデータがたまっていたら、Q-Tableを求めるNNのweightを更新する
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
            epsilon-greedyをやっているだけだが、gpuとのやり取りのために、複雑になっている
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # stateを、行ベクトルにして、GPUに送る
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # 評価モードに切り替える（dropoutなどが無効化される）
        self.qnetwork_local.eval()

        # メモリ節約のために、torch.no_grad()
        with torch.no_grad():
            # .forwar()を実行し、NNの出力（64,1）を得る
            # gpu上のモデルで、gpu上のstateを使って処理する
            # ここで得たactionは、epsilon-greedyで使用される
            action_values = self.qnetwork_local(state)

        # 訓練モードに切り替える（dropoutなどが有効化される）
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
           Q-Tableを求めるNNのweightを更新する
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # state.shape       = [BatchSize, 8]
        # next_states.shape = [BatchSize, 8]
        # actions.shape = [BatchSize, 1]
        # rewards.shape = [BatchSize, 1]
        # dones.shape   = [BatchSize, 1]
        # experiences => 下記5種のデータを保管したtuple
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss

        #----------------- <Pytorch-Tips> -----------------------#
        # nn.Module の __call__ に記載項目は、 インスタンスに引数を与えることで、 __call__　を呼び出せる
        # self.qnetwork_target(next_states)　は self.qnetwork_target.forward(next_states)　と同じ
        #   variable型からtensor型を取り出す際は .detach()　を使う
        # .max(1) col方向に最大値を取得 => 行ベクトルになる => [0]番目の要素
        # pytorchについて : .max()[0] => 最大値を返す
        # pytorchについて : .max()[1] => 最大値に対応したindexを返す(argmax)
        # pytorchについて : .unsqueeze(0) => 行ベクトルにする
        # pytorchについて : .unsqueeze(1) => 列ベクトルにする
        # pytorchについて : .gather(dim, index) => dimで指定した方向に探す、indexで指定したidを出力する
        #--------------------------------------------------------#

        # self.qnetwork_target の forward()関数が実行される、3個のLinear層の計算結果を出力
        # Get max predicted Q values (for next states) from target model
        # Size = [BatchSize, 1]
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        # Size = [BatchSize, 1]
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # Size = [BatchSize, 1]
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # BatchSize分の誤差をまとめて計算し、スカラーを出力
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        # 毎回バックプロパゲーションの初期値をリセット
        self.optimizer.zero_grad()

        # lossの勾配を計算
        loss.backward()

        # 上の勾配を使ってパラメータを更新する
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # ニューラルネットワークの重み全てに対し、一つずつ、soft-updateをしている
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        # ニューラルネットワークの重み全てに対し、一つずつ、soft-updateをしている
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # namedtupleにデータを登録し、Listに追加しているだけ
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # memoryには、最大buffer_size(=1e5)個のデータが入っている
        # その中から、batch_size(=64)個だけ、ランダムに抽出 => experiences
        experiences = random.sample(self.memory, k=self.batch_size)

        # namedtulpleのデータを、それぞれのデータに分配する
        states      = torch.from_numpy(np.vstack([e.state      for e in experiences if e is not None])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences if e is not None])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)