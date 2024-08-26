"""
DynamicLight under feature fusion method 1
Input shape: [batch, max_lane*4]
Created by Liang Zhang
"""
from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Subtract, Add, MultiHeadAttention, Activation, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import copy, os


class DynamicLightAgent(NetworkAgent):

    def build_network(self):
        # 输入层1：表示总特征的输入，形状为 (self.max_lane = 16, self.num_feat1 = 6)
        ins0 = Input(shape=(self.max_lane, self.num_feat1), name="input_total_features") #
        # 输入层2：表示当前相位的输入，形状为 (self.max_lane = 16,)
        ins1 = Input(shape=(self.max_lane,), name="input_cur_phase")

        # 输入层3：表示用于选择相位的矩阵，形状为 (1, self.num_phases = 4)
        ins2 = Input(shape=(1, self.num_phases))

        # 定义嵌入层和输出维度，uni_dim 和 dims2 分别为嵌入维度和隐藏层维度
        uni_dim, dims2 = 4, 16

        # 嵌入层：对输入的相位（ins1）进行嵌入，并使用 sigmoid 激活函数
        #embedding（输入维度，输出维度，多少个输入维度）


        phase_emb = Activation('sigmoid')(Embedding(2, uni_dim, input_length=self.max_lane)(ins1))

        # 将输入的特征 ins0 在最后一个维度上拆分为 num_feat1 个子特征
        #ins0 = <KerasTensor: shape=(None, 16, 6) dtype=float32 (created by layer 'input_total_features')>
        #feat_list = {list:6}[<KerasTensor: shape=(None, 16, 1) dtype=float32 (created by layer 'tf.split')>, <KerasTensor: shape=(None, 16, 1) dtype=float32 (created by layer 'tf.split')>, <KerasTensor: shape=(None, 16, 1) dtype=float32 (created by layer 'tf.split')>, <KerasTensor: shape=(None, 16, 1) dtype=float32 (created by layer 'tf.split')>, <KerasTensor: shape=(None, 16, 1) dtype=float32 (created by layer 'tf.split')>, <KerasTensor: shape=(None, 16, 1) dtype=float32 (created by layer 'tf.split')>]
        feat_list = tf.split(ins0, self.num_feat1, axis=2)

        # 对每个子特征进行嵌入，并使用 sigmoid 激活函数
        # 对应特征向量嵌入的输出是 (self.max_lane, uni_dim)
        feat_embs = [Dense(uni_dim, activation='sigmoid')(feat_list[i]) for i in range(self.num_feat1)]

        # 将相位嵌入添加到特征嵌入列表中
        # ↓feat_embs = {list:7}[<KerasTensor: shape=(None, 16, 4) dtype=float32 (created by layer 'dense')>, <KerasTensor: shape=(None, 16, 4) dtype=float32 (created by layer 'dense_1')>, <KerasTensor: shape=(None, 16, 4) dtype=float32 (created by layer 'dense_2')>, <KerasTensor: shape=(None, 16, 4) dtype=float32 (created by layer 'dense_3')>, <KerasTensor: shape=(None, 16, 4) dtype=float32 (created by layer 'dense_4')>, <KerasTensor: shape=(None, 16, 4) dtype=float32 (created by layer 'dense_5')>, <KerasTensor: shape=(None, 16, 4) dtype=float32 (created by layer 'activation')>]
        feat_embs.append(phase_emb)

        # 将所有特征嵌入在最后一个维度上连接起来，形成新的特征表示
        # ↓<KerasTensor: shape=(None, 16, 28) dtype=float32 (created by layer 'tf.concat')>
        feats = tf.concat(feat_embs, axis=2)

        # 对连接后的特征表示应用全连接层，并使用 ReLU 激活函数进行非线性变换
        # ↓<KerasTensor: shape=(None, 16, 16) dtype=float32 (created by layer 'dense_6')>
        feats = Dense(dims2, activation="relu")(feats)

        # 将结果特征在第一个维度上拆分为 self.max_lane 个 lane 特征
        # ↓{list:16}[<KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'tf.split_1')>]
        lane_feats = tf.split(feats, self.max_lane, axis=1)

        # 存储每个相位的特征
        phase_feats = []

        # 定义多头注意力机制的层，参数 4 是头的数量，16 是每个头的输出维度，attention_axes 指定注意力操作的轴
        MHA1 = MultiHeadAttention(4, 16, attention_axes=1)

        # 使用 Lambda 层创建一个计算平均值的操作，计算沿第一个维度的平均值
        Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))

        # 遍历每个相位，将 lane 特征按 phase_map 中定义的映射进行组合并生成相位特征
        for i in range(self.num_phases):
            # 选择属于当前相位 i 的所有 lane 特征，并在最后一个维度上连接
            # ↓<KerasTensor: shape=(None, 2, 16) dtype=float32 (created by layer 'tf.concat_1')>
            tmp_feat_1 = tf.concat([lane_feats[idx] for idx in self.phase_map[i]], axis=1)

            # 对连接后的 lane 特征应用多头注意力机制
            # ↓<KerasTensor: shape=(None, 2, 16) dtype=float32 (created by layer 'multi_head_attention')>
            tmp_feat_2 = MHA1(tmp_feat_1, tmp_feat_1)

            # 对注意力输出的特征进行平均
            # ↓<KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'lambda')>
            tmp_feat_3 = Mean1(tmp_feat_2)

            # 将计算的相位特征添加到 phase_feats 列表中
            # ↓{list:4}[<KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'lambda')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'lambda')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'lambda')>, <KerasTensor: shape=(None, 1, 16) dtype=float32 (created by layer 'lambda')>]
            phase_feats.append(tmp_feat_3)

        # 将所有相位特征在最后一个维度上连接起来
        # ↓<KerasTensor: shape=(None, 4, 16) dtype=float32 (created by layer 'tf.concat_5')>
        phase_feat_all = tf.concat(phase_feats, axis=1)

        # 对相位特征进行多头注意力操作
        #<KerasTensor: shape=(None, 4, 16) dtype=float32 (created by layer 'multi_head_attention_1')>
        att_encoding = MultiHeadAttention(4, 8, attention_axes=1)(phase_feat_all, phase_feat_all)

        # 对注意力编码的特征应用两层全连接层，并使用 ReLU 激活函数
        hidden = Dense(20, activation="relu")(att_encoding)
        hidden = Dense(20, activation="relu")(hidden)
        #hidden = <KerasTensor: shape=(None, 4, 20) dtype=float32 (created by layer 'dense_8')>

################################################################################################################################
        # 最终输出一个线性激活的特征，作为最终相位特征
        #<KerasTensor: shape=(None, 4, 1) dtype=float32 (created by layer 'beformerge')>
        phase_feature_final = Dense(1, activation="linear", name="beformerge")(hidden)


        # 将线性特征重新调整形状为 (4,) 作为评分输出（pscore）
        #<KerasTensor: shape=(None, 4) dtype=float32 (created by layer 'reshape')>
        pscore = Reshape((4,))(phase_feature_final)
################################################################################################################################


        # 选择当前相位特征，通过 ins2 和相位特征矩阵相乘来选择特征
        #<KerasTensor: shape=(None, 16) dtype=float32 (created by layer 'reshape_1')>
        selected_phase_feat = Lambda(lambda x: tf.matmul(x[0], x[1]))([ins2, phase_feat_all])

        # 将选择后的相位特征调整为 dims2 维度
        #<KerasTensor: shape=(None, 16) dtype=float32 (created by layer 'reshape_1')>
        selected_phase_feat = Reshape((dims2,))(selected_phase_feat)

        # 对选择的特征应用两层全连接层，并使用 ReLU 激活函数
        hidden2 = Dense(20, activation="relu")(selected_phase_feat)
        hidden2 = Dense(20, activation="relu")(hidden2)
        #hidden2 = <KerasTensor: shape=(None, 20) dtype=float32 (created by layer 'dense_10')>

        # 调用 dueling_block 方法处理隐层输出，生成决策评分（dscore）
        #<KerasTensor: shape=(None, 7) dtype=float32 (created by layer 'dueling_q_values')>
        dscore = self.dueling_block(hidden2)
################################################################################################################################
        # 创建 Keras 模型对象，指定输入层和输出层
        # 这里使用了keras模型对象，所以才会无法在代码中找到self.q_network.predict(****)中predict
        network = Model(inputs=[ins0, ins1, ins2],
                        outputs=[pscore, dscore])

        # 编译模型，指定优化器为 Adam，并设置学习率和损失函数
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])

        # 打印模型结构摘要
        network.summary()

        # 返回构建的网络模型
        return network

    def freze_layers(self):
        for layer in self.q_network.layers[0:21]:
            layer.trainable = False
        self.q_network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                               loss=self.dic_agent_conf["LOSS_FUNCTION"])

    def unfreze_layers(self):
        for layer in self.q_network.layers[0:21]:
            layer.trainable = True
        self.q_network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE2"], epsilon=1e-08),
                               loss=self.dic_agent_conf["LOSS_FUNCTION"])

    def resetlr(self):
        self.q_network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE3"], epsilon=1e-08),
                               loss=self.dic_agent_conf["LOSS_FUNCTION"])
    
    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        val_spl = 0.2
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        batch_size = min(self.dic_agent_conf["BATCH_SIZE1"], len(self.Ys[0]))
        self.q_network.fit(self.Xs, self.Ys, batch_size=batch_size, epochs=epochs, shuffle=1,
                           verbose=2, validation_split=val_spl, callbacks=[early_stopping])
        print("=== training two model ===")

    def dueling_block(self, inputs):
        tmp_v = Dense(20, activation="relu", name="dense_values")(inputs)
        value = Dense(1, activation="linear", name="dueling_values")(tmp_v)
        tmp_a = Dense(20, activation="relu", name="dense_a")(inputs)
        a = Dense(self.num_action_dur, activation="linear", name="dueling_advantages")(tmp_a)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantages = Subtract()([a, mean])
        q_values = Add(name='dueling_q_values')([value, advantages])
        return q_values
    
    def choose_action(self, states, list_need):
        dic_state_feature_arrays = {}
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE_1"])
        cur_phase = []

        #used_feature = ['phase_total', 'lane_queue_vehicle_in', 'lane_run_in_part', 'num_in_deg']
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []


        for s in states:
            for feature_name in used_feature:
                if feature_name == "phase_total":
                    cur_phase.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        used_feature.remove("phase_total")

        #剩下的3个特征连接起来'lane_queue_vehicle_in', 'lane_run_in_part', 'num_in_deg'
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), self.max_lane, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)


        # phase action  通过网络预测动作
        tmp_p, _ = self.q_network.predict([state_input, np.array(cur_phase), np.random.rand(len(states),1, 4)])

        #SROUND之前都是随机选择 p action， tmp_p返回的是（196,4）
        if self.cnt_round < self.dic_agent_conf["SROUND"]:
            paction = self.epsilon_choice_one(tmp_p)
        else:
            paction = np.argmax(tmp_p, axis=1)


        # duration action
        #将动作调整一下维度
        phase_idx =  np.array(paction).reshape(len(paction), 1, 1)#维度(196, 1, 1)
        phase_matrix = self.phase_index2matrix(phase_idx)#维度(196, 1, 4)，

        #使用选择的相位来和一些特征来决定持续时间 state_input 维度(196, 16, 6)
        _, tmp_d = self.q_network.predict([state_input, np.array(cur_phase), phase_matrix])
        if self.cnt_round < self.dic_agent_conf["SROUND"]:

            #注意：这里80回合之内都是固定的 duration，所以为了方便理解逻辑我把这里注释改掉了
            # daction = [1] * len(states)
            daction = np.argmax(tmp_d, axis=1)


        #这里进入了SROUND2回合后，网络输出的维度会变化，一开始是196,后来30 42 等数字，未查明是故意设计还是错误
        elif self.cnt_round < self.dic_agent_conf["SROUND2"]:
            daction = self.epsilon_choice_two(tmp_d)
        else:
            daction = np.argmax(tmp_d, axis=1)
        return paction, daction

    def phase_index2matrix(self, phase_index):
        # [batch, 1] -> [batch, 1, num_phase]
        lab = to_categorical(phase_index, num_classes=self.num_phases)
        return lab
    
    def epsilon_choice_one(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_phases, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act
    
    def epsilon_choice_two(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_action_dur, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act
    
    def prepare_Xs_Y(self, memory):
        """ used for update phase control model """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))
        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE1"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE_1"][1:]
        _phase1, _phase2 = [], []
        _state = [[] for _ in used_feature]
        _next_state = [[] for _ in used_feature]
        _action1 = []
        _action2 = []
        _action3 = []
        _reward = []
        for i in range(len(sample_slice)):
            state, action1, action2, next_state, reward, _ = sample_slice[i]
            for feat_idx, feat_name in enumerate(used_feature):
                _state[feat_idx].append(state[feat_name])
                _next_state[feat_idx].append(next_state[feat_name])
            _phase1.append(state["phase_total"])
            _phase2.append(next_state["phase_total"])
            _action1.append([[action1]])
            _action3.append(action1)
            _action2.append(action2)
            _reward.append(reward)
        _state2 = np.concatenate([np.array(ss).reshape(len(ss), self.max_lane, -1) for ss in _state], axis=-1)
        _next_state2 = np.concatenate([np.array(ss).reshape(len(ss), self.max_lane, -1) for ss in _next_state], axis=-1)
        phase_matrix = self.phase_index2matrix(np.array(_action1))
        
        cur_p, cur_d = self.q_network.predict([_state2, np.array(_phase1), phase_matrix])
        next_p, next_d = self.q_network_bar.predict([_next_state2, np.array(_phase2), phase_matrix])
        
        target1 = np.copy(cur_p)
        target2 = np.copy(cur_d)

        #计算y，每达到SROUND之前，只更新phase部分，将目标函数y计算出来使用fit更新，而tatget2使用的是现在的Q所以不用更新loss就是0，从而达到不更新target2的目的
        if self.cnt_round < self.dic_agent_conf["SROUND"]:
            for i in range(len(sample_slice)):
                target1[i, _action3[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * np.max(next_p[i, :])
        elif self.cnt_round < self.dic_agent_conf["SROUND2"]:
            for i in range(len(sample_slice)):
                target2[i, _action2[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * np.max(next_d[i, :])
        else:
            for i in range(len(sample_slice)):
                target1[i, _action3[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] *np.max(next_p[i, :])
            for i in range(len(sample_slice)):
                target2[i, _action2[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * np.max(next_d[i, :])
            
        self.Xs = [_state2, np.array(_phase1), phase_matrix]
        self.Ys = [target1, target2]

    def save_network(self, file_name):
        if self.cnt_round == self.dic_agent_conf["SROUND"]-1:
            self.freze_layers()
        if self.cnt_round == self.dic_agent_conf["SROUND2"]-1:
            self.unfreze_layers()
        if self.cnt_round == self.dic_agent_conf["SROUND3"]-1:
            self.resetlr()
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))