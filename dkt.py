# %%
from __future__ import print_function
import random
import time
import sys
import os
import datetime

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import tensorflow as tf


# %%

# 配置config
class TrainConfig(object):
    epochs = 16
    decay_rate = 0.90
    learning_rate = 0.001
    evaluate_every = 100
    checkpoint_every = 100
    max_grad_norm = 3.0


class ModelConfig(object):
    hidden_layers = [200]
    dropout_keep_prob = 0.6


class Config(object):
    batch_size = 32
    num_skills = 123
    input_size = num_skills * 2

    trainConfig = TrainConfig()
    modelConfig = ModelConfig()


# 实例化config
config = Config()


# %%

# 生成数据
class DataGenerator(object):
    # 导入的seqs是train_seqs，或者是test_seqs
    def __init__(self, fileName, config):
        self.fileName = fileName
        self.train_seqs = []
        self.test_seqs = []
        self.infer_seqs = []
        self.batch_size = config.batch_size
        self.pos = 0
        self.end = False
        self.num_skills = config.num_skills
        self.skills_to_int = {}  # 知识点到索引的映射
        self.int_to_skills = {}  # 索引到知识点的映射

    def read_file(self):
        # 从文件中读取数据，返回读取出来的数据和知识点个数
        # 保存每个学生的做题信息 {学生id: [[知识点id，答题结果], [知识点id，答题结果], ...]}，用一个二元列表来表示一个学生的答题信息
        seqs_by_student = {}
        skills = []  # 统计知识点的数量，之后输入的向量长度就是两倍的知识点数量
        count = 0
        with open(self.fileName, 'r') as f:
            for line in f:
                fields = line.strip().split(" ")  # 一个列表，[学生id，知识点id，答题结果]
                student, skill, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
                skills.append(skill)  # skill实际上是用该题所属知识点来表示的
                seqs_by_student[student] = seqs_by_student.get(student, []) + [[skill, is_correct]]  # 保存每个学生的做题信息
        return seqs_by_student, list(set(skills))

    def gen_dict(self, unique_skills):
        """
        构建知识点映射表，将知识点id映射到[0, 1, 2...]表示
        :param unique_skills: 无重复的知识点列表
        :return:
        """
        sorted_skills = sorted(unique_skills)
        skills_to_int = {}
        int_to_skills = {}
        for i in range(len(sorted_skills)):
            skills_to_int[sorted_skills[i]] = i
            int_to_skills[i] = sorted_skills[i]

        self.skills_to_int = skills_to_int
        self.int_to_skills = int_to_skills

    def split_dataset(self, seqs_by_student, sample_rate=0.2, random_seed=1):
        # 将数据分割成测试集和训练集
        sorted_keys = sorted(seqs_by_student.keys())  # 得到排好序的学生id的列表

        random.seed(random_seed)
        # 随机抽取学生id，将这部分学生作为测试集
        test_keys = set(random.sample(sorted_keys, int(len(sorted_keys) * sample_rate)))

        # 此时是一个三层的列表来表示的，最外层的列表中的每一个列表表示一个学生的做题信息
        test_seqs = [seqs_by_student[k] for k in seqs_by_student if k in test_keys]
        train_seqs = [seqs_by_student[k] for k in seqs_by_student if k not in test_keys]
        return train_seqs, test_seqs

    def gen_attr(self, is_infer=False):
        """
        生成待处理的数据集
        :param is_infer: 判断当前是训练模型还是利用模型进行预测
        :return:
        """
        if is_infer:
            seqs_by_students, skills = self.read_file()
            self.infer_seqs = seqs_by_students
        else:
            seqs_by_students, skills = self.read_file()
            train_seqs, test_seqs = self.split_dataset(seqs_by_students)
            self.train_seqs = train_seqs
            self.test_seqs = test_seqs

        self.gen_dict(skills)  # 生成知识点到索引的映射字典

    def pad_sequences(self, sequences, maxlen=None, value=0.):
        # 按每个batch中最长的序列进行补全, 传入的sequences是二层列表
        # 统计一个batch中每个序列的长度，其实等于seqs_len
        lengths = [len(s) for s in sequences]
        # 统计下该batch中序列的数量
        nb_samples = len(sequences)
        # 如果没有传入maxlen参数就自动获取最大的序列长度
        if maxlen is None:
            maxlen = np.max(lengths)
        # 构建x矩阵
        x = (np.ones((nb_samples, maxlen)) * value).astype(np.int32)

        # 遍历batch，去除每一个序列
        for idx, s in enumerate(sequences):
            trunc = np.asarray(s, dtype=np.int32)
            x[idx, :len(trunc)] = trunc

        return x

    def num_to_one_hot(self, num, dim):
        # 将题目转换成one-hot的形式， 其中dim=num_skills * 2，前半段表示错误，后半段表示正确
        base = np.zeros(dim)
        if num >= 0:
            base[num] += 1
        return base

    def format_data(self, seqs):
        # 生成输入数据和输出数据，输入数据是每条序列的前n-1个元素，输出数据是每条序列的后n-1个元素

        # 统计一个batch_size中每条序列的长度，在这里不对序列固定长度，通过条用tf.nn.dynamic_rnn让序列长度可以不固定
        seq_len = np.array(list(map(lambda seq: len(seq) - 1, seqs)))
        max_len = max(seq_len)  # 获得一个batch_size中最大的长度
        # i表示第i条数据，j只从0到len(i)-1，x作为输入只取前len(i)-1个，sequences=[j[0] + num_skills * j[1], ....]
        # 此时要将知识点id j[0] 转换成index表示
        x_sequences = np.array([[(self.skills_to_int[j[0]] + self.num_skills * j[1]) for j in i[:-1]] for i in seqs])
        # 将输入的序列用-1进行补全，补全后的长度为当前batch的最大序列长度
        x = self.pad_sequences(x_sequences, maxlen=max_len, value=-1)

        # 构建输入值input_x，x为一个二层列表，i表示一个学生的做题信息，也就是一个序列，j就是一道题的信息
        input_x = np.array([[self.num_to_one_hot(j, self.num_skills * 2) for j in i] for i in x])

        # 遍历batch_size，然后取每条序列的后len(i)-1 个元素中的知识点id为target_id
        target_id_seqs = np.array([[self.skills_to_int[j[0]] for j in i[1:]] for i in seqs])
        target_id = self.pad_sequences(target_id_seqs, maxlen=max_len, value=0)

        # 同target_id
        target_correctness_seqs = np.array([[j[1] for j in i[1:]] for i in seqs])
        target_correctness = self.pad_sequences(target_correctness_seqs, maxlen=max_len, value=0)

        return dict(input_x=input_x, target_id=target_id, target_correctness=target_correctness,
                    seq_len=seq_len, max_len=max_len)

    def next_batch(self, seqs):
        # 接收一个序列，生成batch

        length = len(seqs)
        num_batchs = length // self.batch_size
        start = 0
        for i in range(num_batchs):
            batch_seqs = seqs[start: start + self.batch_size]
            start += self.batch_size
            params = self.format_data(batch_seqs)

            yield params


fileName = "./data/assistments.txt"
dataGen = DataGenerator(fileName, config)
dataGen.gen_attr()

# %%

train_seqs = dataGen.train_seqs
params = next(dataGen.next_batch(train_seqs))
print("skill num: {}".format(len(dataGen.skills_to_int)))
print("train_seqs length: {}".format(len(dataGen.train_seqs)))
print("test_seqs length: {}".format(len(dataGen.test_seqs)))
print("input_x shape: {}".format(params['input_x'].shape))
print(params["input_x"][1][0])


# %%

# 构建模型
class TensorFlowDKT(object):
    def __init__(self, config):
        # 导入配置好的参数
        self.hiddens = hiddens = config.modelConfig.hidden_layers
        self.num_skills = num_skills = config.num_skills
        self.input_size = input_size = config.input_size
        self.batch_size = batch_size = config.batch_size
        self.keep_prob_value = config.modelConfig.dropout_keep_prob

        # 定义需要喂给模型的参数
        self.max_steps = tf.placeholder(tf.int32, name="max_steps")  # 当前batch中最大序列长度
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name="input_x")

        self.sequence_len = tf.placeholder(tf.int32, [batch_size], name="sequence_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout keep prob

        self.target_id = tf.placeholder(tf.int32, [batch_size, None], name="target_id")
        self.target_correctness = tf.placeholder(tf.float32, [batch_size, None], name="target_correctness")
        self.flat_target_correctness = None

        # 构建lstm模型结构
        hidden_layers = []
        for idx, hidden_size in enumerate(hiddens):
            lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            hidden_layer = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_layer,
                                                         output_keep_prob=self.keep_prob)
            hidden_layers.append(hidden_layer)
        self.hidden_cell = tf.nn.rnn_cell.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

        # 采用动态rnn，动态输入序列的长度
        outputs, self.current_state = tf.nn.dynamic_rnn(cell=self.hidden_cell,
                                                        inputs=self.input_data,
                                                        sequence_length=self.sequence_len,
                                                        dtype=tf.float32)

        # 隐层到输出层的权重系数[最后隐层的神经元数量，知识点数]
        output_w = tf.get_variable("W", [hiddens[-1], num_skills])
        output_b = tf.get_variable("b", [num_skills])

        self.output = tf.reshape(outputs, [batch_size * self.max_steps, hiddens[-1]])
        # 因为权值共享的原因，对生成的矩阵[batch_size * self.max_steps, num_skills]中的每一行都加上b
        self.logits = tf.matmul(self.output, output_w) + output_b

        self.mat_logits = tf.reshape(self.logits, [batch_size, self.max_steps, num_skills])

        # 对每个batch中每个序列中的每个时间点的输出中的每个值进行sigmoid计算，这里的值表示对某个知识点的掌握情况，
        # 每个时间点都会输出对所有知识点的掌握情况
        self.pred_all = tf.sigmoid(self.mat_logits, name="pred_all")

        # 计算损失loss
        flat_logits = tf.reshape(self.logits, [-1])

        flat_target_correctness = tf.reshape(self.target_correctness, [-1])
        self.flat_target_correctness = flat_target_correctness

        flat_base_target_index = tf.range(batch_size * self.max_steps) * num_skills

        # 因为flat_logits的长度为batch_size * num_steps * num_skills，我们要根据每一步的target_id将其长度变成batch_size * num_steps
        flat_base_target_id = tf.reshape(self.target_id, [-1])

        flat_target_id = flat_base_target_id + flat_base_target_index
        # gather是从一个tensor中切片一个子集
        flat_target_logits = tf.gather(flat_logits, flat_target_id)

        # 对切片后的数据进行sigmoid转换
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [batch_size, self.max_steps]), name="pred")
        # 将sigmoid后的值表示为0或1
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.float32, name="binary_pred")

        # 定义损失函数
        with tf.name_scope("loss"):
            # flat_target_logits_sigmoid = tf.nn.log_softmax(flat_target_logits)
            # self.loss = -tf.reduce_mean(flat_target_correctness * flat_target_logits_sigmoid)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness,
                                                                               logits=flat_target_logits))


# %%

# 训练模型
def mean(item):
    return sum(item) / len(item)


def gen_metrics(sequence_len, binary_pred, pred, target_correctness):
    """
    生成auc和accuracy的指标值
    :param sequence_len: 每一个batch中各序列的长度组成的列表
    :param binary_pred:
    :param pred:
    :param target_correctness:
    :return:
    """
    binary_preds = []
    preds = []
    target_correctnesses = []
    for seq_idx, seq_len in enumerate(sequence_len):
        binary_preds.append(binary_pred[seq_idx, :seq_len])
        preds.append(pred[seq_idx, :seq_len])
        target_correctnesses.append(target_correctness[seq_idx, :seq_len])

    new_binary_pred = np.concatenate(binary_preds)
    new_pred = np.concatenate(preds)
    new_target_correctness = np.concatenate(target_correctnesses)
    with tf.name_scope('auc'):
        auc = roc_auc_score(new_target_correctness, new_pred)

    with tf.name_scope('accuracy'):
        accuracy = accuracy_score(new_target_correctness, new_binary_pred)

    with tf.name_scope('precision'):
        precision = precision_score(new_target_correctness, new_binary_pred)

    with tf.name_scope('recall'):
        recall = recall_score(new_target_correctness, new_binary_pred)


    return auc, accuracy, precision, recall


# %%

class DKTEngine(object):

    def __init__(self):
        self.config = Config()
        self.train_dkt = None
        self.test_dkt = None
        self.sess = None
        self.global_step = 0

    def add_gradient_noise(self, grad, stddev=1e-3, name=None):
        """
        Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
        """
        with tf.op_scope([grad, stddev], name, "add_gradient_noise") as name:
            grad = tf.convert_to_tensor(grad, name="grad")
            gn = tf.random_normal(tf.shape(grad), stddev=stddev)
            return tf.add(grad, gn, name=name)

    def train_step(self, params, train_op, train_summary_op, train_summary_writer):
        """
        A single training step
        """
        dkt = self.train_dkt
        sess = self.sess
        global_step = self.global_step

        feed_dict = {dkt.input_data: params['input_x'],
                     dkt.target_id: params['target_id'],
                     dkt.target_correctness: params['target_correctness'],
                     dkt.max_steps: params['max_len'],
                     dkt.sequence_len: params['seq_len'],
                     dkt.keep_prob: self.config.modelConfig.dropout_keep_prob}

        _, step, summaries, loss, binary_pred, pred, target_correctness = sess.run(
            [train_op, global_step, train_summary_op, dkt.loss, dkt.binary_pred, dkt.pred, dkt.target_correctness],
            feed_dict)

        auc, accuracy, precision, recall = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)
        with tf.name_scope('auc'):
            aucc =auc
        with tf.name_scope('accuracy'):
            accur = accuracy
        with tf.name_scope('precision'):
            prec = precision
        with tf.name_scope('recall'):
            reca = recall

        time_str = datetime.datetime.now().isoformat()
        print("train: {}: step {}, loss {}, acc {}, auc: {}, precision: {}, recall: {}".format(time_str, step, loss,
                                                                                               accuracy,
                                                                                               auc, precision, recall))
        train_summary_writer.add_summary(summaries, step)
        train_summary_writer.add_summary(summaries, step)

    def dev_step(self, params, dev_summary_op, writer=None):
        """
        Evaluates model on a dev set
        """
        dkt = self.test_dkt
        sess = self.sess
        global_step = self.global_step

        feed_dict = {dkt.input_data: params['input_x'],
                     dkt.target_id: params['target_id'],
                     dkt.target_correctness: params['target_correctness'],
                     dkt.max_steps: params['max_len'],
                     dkt.sequence_len: params['seq_len'],
                     dkt.keep_prob: 1.0}
        step, summaries, loss, pred, binary_pred, target_correctness = sess.run(
            [global_step, dev_summary_op, dkt.loss, dkt.pred, dkt.binary_pred, dkt.target_correctness],
            feed_dict)

        auc, accuracy, precision, recall = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)

        if writer:
            writer.add_summary(summaries, step)

        return loss, accuracy, auc, precision, recall

    def run_epoch(self, fileName):
        """
        训练模型
        :param filePath:
        :return:
        """

        # 实例化配置参数对象
        config = Config()

        # 实例化数据生成对象
        dataGen = DataGenerator(fileName, config)
        dataGen.gen_attr()  # 生成训练集和测试集

        train_seqs = dataGen.train_seqs
        test_seqs = dataGen.test_seqs

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        self.sess = sess

        with sess.as_default():
            # 实例化dkt模型对象
            with tf.name_scope("train"):
                with tf.variable_scope("dkt", reuse=None):
                    train_dkt = TensorFlowDKT(config)

            with tf.name_scope("test"):
                with tf.variable_scope("dkt", reuse=True):
                    test_dkt = TensorFlowDKT(config)

            self.train_dkt = train_dkt
            self.test_dkt = test_dkt

            global_step = tf.Variable(0, name="global_step", trainable=False)
            self.global_step = global_step

            # 定义一个优化器
            optimizer = tf.train.AdamOptimizer(config.trainConfig.learning_rate)
            grads_and_vars = optimizer.compute_gradients(train_dkt.loss)

            # 对梯度进行截断，并且加上梯度噪音
            grads_and_vars = [(tf.clip_by_norm(g, config.trainConfig.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]
            # grads_and_vars = [(self.add_gradient_noise(g), v) for g, v in grads_and_vars]

            # 定义图中最后的节点
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            # 保存各种变量或结果的值
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("writing to {}".format(out_dir))

            # 训练时的 Summaries
            train_loss_summary = tf.summary.scalar("loss", train_dkt.loss)
            # recall_summary = tf.summary.scalar('recall', train_dkt.recall)
            # precision_summary = tf.summary.scalar('precision', train_dkt.precision)
            # auc_summary = tf.summary.scalar('auc', train_dkt.auc)
            # accuracy_summary = tf.summary.scalar('accuracy',train_dkt. accuracy)
            train_summary_op = tf.summary.merge([train_loss_summary,grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # 测试时的 summaries
            test_loss_summary = tf.summary.scalar("loss", test_dkt.loss)
            dev_summary_op = tf.summary.merge([test_loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())

            print("初始化完毕，开始训练")
            for i in range(config.trainConfig.epochs):
                np.random.shuffle(train_seqs)
                for params in dataGen.next_batch(train_seqs):
                    # 批次获得训练集，训练模型
                    self.train_step(params, train_op, train_summary_op, train_summary_writer)

                    current_step = tf.train.global_step(sess, global_step)
                    # train_step.run(feed_dict={x: batch_train[0], y_actual: batch_train[1], keep_prob: 0.5})
                    # 对结果进行记录
                    if current_step % config.trainConfig.evaluate_every == 0:
                        print("\nEvaluation:")
                        # 获得测试数据

                        losses = []
                        accuracys = []
                        aucs = []
                        precisions = []
                        recalls = []
                        for params in dataGen.next_batch(test_seqs):
                            loss, accuracy, auc, precision, recall = self.dev_step(params, dev_summary_op, writer=None)
                            losses.append(loss)
                            accuracys.append(accuracy)
                            aucs.append(auc)
                            precisions.append(precision)
                            recalls.append(recall)

                        time_str = datetime.datetime.now().isoformat()
                        print("dev: {}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".
                              format(time_str, current_step, mean(losses), mean(accuracys), mean(aucs),
                                     mean(precisions), mean(recalls)))

                    if current_step % config.trainConfig.checkpoint_every == 0:
                        path = saver.save(sess, "model/my-model", global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    fileName = "./data/assistments.txt"
    dktEngine = DKTEngine()
    dktEngine.run_epoch(fileName)


# %%

# 模型预测
def load_model(fileName):
    # 实例化配置参数对象
    config = Config()

    # 实例化数据生成对象
    dataGen = DataGenerator(fileName, config)
    dataGen.gen_attr()  # 生成训练集和测试集

    test_seqs = dataGen.test_seqs

    with tf.Session() as sess:
        accuracys = []
        aucs = []
        step = 1

        for params in dataGen.next_batch(test_seqs):
            print("step: {}".format(step))

            saver = tf.train.import_meta_graph("model/my-model-800.meta")
            saver.restore(sess, tf.train.latest_checkpoint("model/"))

            # 获得默认的计算图结构
            graph = tf.get_default_graph()

            # 获得需要喂给模型的参数，输出的结果依赖的输入值
            input_x = graph.get_operation_by_name("test/dkt/input_x").outputs[0]
            target_id = graph.get_operation_by_name("test/dkt/target_id").outputs[0]
            keep_prob = graph.get_operation_by_name("test/dkt/keep_prob").outputs[0]
            max_steps = graph.get_operation_by_name("test/dkt/max_steps").outputs[0]
            sequence_len = graph.get_operation_by_name("test/dkt/sequence_len").outputs[0]

            # 获得输出的结果
            pred_all = graph.get_tensor_by_name("test/dkt/pred_all:0")
            pred = graph.get_tensor_by_name("test/dkt/pred:0")
            binary_pred = graph.get_tensor_by_name("test/dkt/binary_pred:0")

            target_correctness = params['target_correctness']
            pred_all, pred, binary_pred = sess.run([pred_all, pred, binary_pred],
                                                   feed_dict={input_x: params["input_x"],
                                                              target_id: params["target_id"],
                                                              keep_prob: 1.0,
                                                              max_steps: params["max_len"],
                                                              sequence_len: params["seq_len"]})

            auc, acc, precision, recall = gen_metrics(params["seq_len"], binary_pred, pred, target_correctness)
            print(auc, acc)
            accuracys.append(acc)
            aucs.append(auc)
            step += 1

        aucMean = mean(aucs)
        accMean = mean(accuracys)

        print("inference  auc: {}  acc: {}".format(aucMean, accMean))


if __name__ == "__main__":
    fileName = "./data/assistments.txt"
    load_model(fileName)
