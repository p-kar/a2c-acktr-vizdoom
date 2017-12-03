import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit
from torchvision import models
from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian

# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release.
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "CNNPolicy" or classname == "MLPPolicy":
        return
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space_shape):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear = nn.Linear(512, 1)

        num_outputs = action_space_shape
        self.dist = Categorical(512, num_outputs)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        return self.critic_linear(x), x

    def get_probs(self, inputs):
        value, x = self(inputs)
        x = self.dist(x)
        probs = F.softmax(x)
        return probs


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space_shape):
        super(MLPPolicy, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)

        num_outputs = action_space_shape
        self.dist = Categorical(64, num_outputs)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x


# Implements the Attend, Adapt and Transfer architecture from ICLR 2017
class A2TPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space_shape, source_models):
        super(A2TPolicy, self).__init__()
        
        # adding all pretrained models to the network
        self.num_source_models = len(source_models)
        for sm, i in zip(source_models, range(len(source_models))):
            self.add_module('source_' + str(i), sm)
        # freezing params for pretrained models
        for param in self.parameters():
            param.requires_grad = False

        # parameters for the base network
        self.base_conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4)
        self.base_conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.base_linear1 = nn.Linear(32 * 9 * 9, 256)

        self.base_dist_linear = nn.Linear(256, action_space_shape)

        self.base_critic_linear = nn.Linear(256, 1)

        # parameters for the attention network
        self.attention_conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4)
        self.attention_conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.attention_linear1 = nn.Linear(32 * 9 * 9, 256)
        
        num_source_models = len(source_models)
        self.attention_dist_linear = nn.Linear(256, num_source_models + 1)

        # sets the module to train mode
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        nninit.xavier_normal(self.base_conv1.weight)
        nninit.constant(self.base_conv1.bias, 0.1)
        nninit.xavier_normal(self.base_conv2.weight)
        nninit.constant(self.base_conv2.bias, 0.1)
        nninit.xavier_normal(self.base_linear1.weight)
        nninit.constant(self.base_linear1.bias, 0.1)
        nninit.xavier_normal(self.base_dist_linear.weight)
        nninit.constant(self.base_dist_linear.bias, 0.1)
        nninit.xavier_normal(self.base_critic_linear.weight)
        nninit.constant(self.base_critic_linear.bias, 0.1)
        nninit.xavier_normal(self.attention_conv1.weight)
        nninit.constant(self.attention_conv1.bias, 0.1)
        nninit.xavier_normal(self.attention_conv2.weight)
        nninit.constant(self.attention_conv2.bias, 0.1)
        nninit.xavier_normal(self.attention_linear1.weight)
        nninit.constant(self.attention_linear1.bias, 0.1)
        nninit.xavier_normal(self.attention_dist_linear.weight)
        nninit.constant(self.attention_dist_linear.bias, 0.1)

    def forward(self, inputs):
        # go forward in the base network
        x = self.base_conv1(inputs)
        x = F.relu(x)

        x = self.base_conv2(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 9 * 9)
        x = self.base_linear1(x)
        x = F.relu(x)

        value = self.base_critic_linear(x)

        x = self.base_dist_linear(x)
        x = F.softmax(x)

        # go forward in the attention network
        y = self.attention_conv1(inputs)
        y = F.relu(y)

        y = self.attention_conv2(y)
        y = F.relu(y)

        y = y.view(-1, 32 * 9 * 9)
        y = self.attention_linear1(y)
        y = F.relu(y)

        y = self.attention_dist_linear(y)
        y = F.softmax(y)

        # combine base and source task outputs
        # with the attention network weights
        source_probs = [getattr(self, 'source_%d' % i).get_probs(inputs)
                        for i in range(self.num_source_models)]
        source_probs.append(x)
        # stacking probability distribution as rows
        z = torch.stack(tuple(source_probs), 0)
        # multiplying by the attentions weights
        y = torch.t(y).unsqueeze(2)
        z = torch.mul(y, z)
        # summing the rows and reshaping as a row
        z = torch.sum(z, 0)

        return value, z

    def act(self, inputs, deterministic=False):
        value, probs = self(inputs)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1)[1]
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, probs = self(inputs)

        log_probs = torch.log(probs)

        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        
        return value, action_log_probs, dist_entropy

    def get_probs(self, inputs):
        value, probs = self(inputs)
        return probs

# changed beginning network to resnet
class ResnetPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space_shape):
        super(ResnetPolicy, self).__init__()

        net = models.resnet18()

        self.resnet = nn.Sequential(net.conv1, 
                        net.bn1, \
                        net.relu, \
                        net.maxpool, \
                        net.layer1, \
                        net.layer2, \
                        net.layer3, \
                        net.layer4, \
                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True))

        self.dist_linear = nn.Linear(512, action_space_shape)
        self.critic_linear = nn.Linear(512, 1)

        # sets the module to train mode
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.dist_linear.weight.data.mul_(relu_gain)
        self.critic_linear.weight.data.mul_(relu_gain)

    def forward(self, inputs):
        # go forward in the base network
        x = self.resnet(inputs)
        x = x.view(-1, 512)

        value = self.critic_linear(x)

        x = self.dist_linear(x)

        return value, x

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        probs = F.softmax(x)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1)[1]
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)

        log_probs = F.log_softmax(x)
        probs = F.softmax(x)

        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        
        return value, action_log_probs, dist_entropy

    def get_probs(self, inputs):
        value, x = self(inputs)
        probs = F.softmax(x)
        return probs

