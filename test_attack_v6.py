
import tensorflow as tf
import numpy as np
import time
import random
import os
from numpy import linalg as LA

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2

# L0 sneaking attack
from l2_sidechannel_attack_v4 import LADMML2re
# L2 sneaking attack
from l2_sidechannel_attack_v5 import LADMML2re

from PIL import Image


def show(img, name = "output.png"):
    fig = (img + 0.5)*255
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)


def generate_data(data, model, samples, targeted=True, target_num=9, start=0, inception=False, seed=3, handpick=False ):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    random.seed(seed)
    inputs = []
    targets = []
    labels = []
    true_ids = []
    sample_set = []

    data_d = data.test_data
    labels_d = data.test_labels

    if handpick:
        if inception:
            deck = list(range(0, 1500))
        else:
            deck = list(range(0, 10000))
        random.shuffle(deck)
        print('Handpicking')

        while (len(sample_set) < samples):
            rand_int = deck.pop()
            pred = model.model.predict(data_d[rand_int:rand_int + 1])

            if inception:
                pred = np.reshape(pred, (labels_d[0:1].shape))

            if (np.argmax(pred, 1) == np.argmax(labels_d[rand_int:rand_int + 1], 1)):
                sample_set.append(rand_int)
        print('Handpicked')
    else:
        sample_set = random.sample(range(0, 10000), samples)

    for i in sample_set:
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), target_num)
            else:
                seq = range(labels_d.shape[1])

            for j in seq:
                if (j == np.argmax(labels_d[start + i])) and (inception == False):
                    continue
                inputs.append(data_d[start + i])
                targets.append(np.eye(labels_d.shape[1])[j])
                labels.append(labels_d[start + i])
                true_ids.append(start + i)
        else:
            inputs.append(data_d[start + i])
            targets.append(labels_d[start + i])
            labels.append(labels_d[start + i])
            true_ids.append(start + i)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)
    return inputs, targets, labels, true_ids


def generate_data_ST(data, model, samples, samplesT, targeted=True, target_num=9, start=0, inception=False, seed=3, handpick=False ):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    random.seed(seed)
    inputs = []
    targets = []
    labels = []
    true_ids = []
    sample_set = []

    data_d = data.test_data
    labels_d = data.test_labels

    if handpick:
        if inception:
            deck = list(range(0, 1500))
        else:
            deck = list(range(0, 10000))
        random.shuffle(deck)
        print('Handpicking')

        while (len(sample_set) < samplesT):
            rand_int = deck.pop()
            pred = model.model.predict(data_d[rand_int:rand_int + 1])

            if inception:
                pred = np.reshape(pred, (labels_d[0:1].shape))

            if (np.argmax(pred, 1) == np.argmax(labels_d[rand_int:rand_int + 1], 1)):
                sample_set.append(rand_int)
        print('Handpicked')
    else:
        sample_set = random.sample(range(0, 10000), samplesT)

    for j, i in enumerate(sample_set):
        if j < samples:
            if targeted:

                seq = np.random.randint(labels_d.shape[1])
                while seq == np.argmax(labels_d[start + i]):
                    seq = np.random.randint(labels_d.shape[1])

                inputs.append(data_d[start + i])
                targets.append(np.eye(labels_d.shape[1])[seq])
                labels.append(labels_d[start + i])
                true_ids.append(start + i)

            else:
                inputs.append(data_d[start + i])
                targets.append(labels_d[start + i])
                labels.append(labels_d[start + i])
                true_ids.append(start + i)
        else:
            inputs.append(data_d[start + i])
            targets.append(labels_d[start + i])
            labels.append(labels_d[start + i])
            true_ids.append(start + i)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)
    return inputs, targets, labels, true_ids

def evaluate_perturbation(args, sess, model, inputs):
    layernum = args['layer_number']
    batch_size = args['batch_size']
    if args['use_kernel']:
        aaa = model.model.layers[layernum].kernel
    else:
        aaa = model.model.layers[layernum].bias
    timg = tf.Variable(np.zeros(inputs.shape), dtype=tf.float32)
    tdelt = tf.Variable(np.zeros(aaa.shape, dtype=np.float32))
    tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

    assign_tdelt = tf.placeholder(tf.float32, aaa.shape)
    assign_timg = tf.placeholder(tf.float32, inputs.shape)
    assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))

    if args['use_kernel']:
        model.model.layers[layernum].kernel = tdelt + model.model.layers[layernum].kernel
    else:
        model.model.layers[layernum].bias = tdelt + model.model.layers[layernum].bias
    output = model.predict(timg)

    if args['use_kernel']:
        model.model.layers[layernum].kernel = model.model.layers[layernum].kernel - tdelt
        bbb = model.model.layers[layernum].kernel
    else:
        model.model.layers[layernum].bias = model.model.layers[layernum].bias - tdelt
        bbb = model.model.layers[layernum].bias
    # these are the variables to initialize when we run
    setup = []
    setup.append(timg.assign(assign_timg))
    setup.append(tlab.assign(assign_tlab))
    setup.append(tdelt.assign(assign_tdelt))

    def doit(inputs, targets, adv):
        batch = inputs[:batch_size]
        batchlab = targets[:batch_size]

        sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tdelt: adv, })
  #      aaaa, bbbb, cccc = sess.run([aaa, bbb, ccc])
  #      print(LA.norm(aaaa - bbbb))
  #      print(LA.norm(aaaa - cccc))
        scores, _, _ = sess.run([output, aaa, bbb])

        return scores

    return doit

def evaluate_perturbation_kb(args, sess, model, inputs):
    layernum = args['layer_number']
    batch_size = inputs.shape[0]
    aaa = model.model.layers[layernum].kernel
    aaa2 = model.model.layers[layernum].bias
    timg = tf.Variable(np.zeros(inputs.shape), dtype=tf.float32)
    tdelt_kernel = tf.Variable(np.zeros(aaa.shape, dtype=np.float32))
    tdelt_bias = tf.Variable(np.zeros(aaa2.shape, dtype=np.float32))
    tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

    assign_tdelt_kernel = tf.placeholder(tf.float32, aaa.shape)
    assign_tdelt_bias = tf.placeholder(tf.float32, aaa2.shape)
    assign_timg = tf.placeholder(tf.float32, inputs.shape)
    assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))

    model.model.layers[layernum].kernel = tdelt_kernel + model.model.layers[layernum].kernel
    model.model.layers[layernum].bias = tdelt_bias + model.model.layers[layernum].bias

    bbb = model.model.layers[layernum].kernel
    output = model.predict(timg)
    l2dist_real = tf.reduce_sum(tf.square(tdelt_kernel)) + tf.reduce_sum(tf.square(tdelt_bias))
    l2dist_real = tf.sqrt(l2dist_real)
    # model.model.save('models/mnist_mod')

    # these are the variables to initialize when we run
    setup = []
    setup.append(timg.assign(assign_timg))
    setup.append(tlab.assign(assign_tlab))
    setup.append(tdelt_kernel.assign(assign_tdelt_kernel))
    setup.append(tdelt_bias.assign(assign_tdelt_bias))

    def doit(inputs, targets, adv):
        batch = inputs[:batch_size]
        batchlab = targets[:batch_size]
        layernum = args['layer_number']
        akernel = model.model.layers[layernum].kernel
        abias = model.model.layers[layernum].bias
        z1 = adv[0: akernel.shape[0] * akernel.shape[1]]
        z2 = adv[akernel.shape[0] * akernel.shape[1]:]
        z1 = np.reshape(z1, akernel.shape)
        z2 = np.reshape(z2, abias.shape)
        sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tdelt_kernel: z1, assign_tdelt_bias: z2,})
        aaaa, bbbb = sess.run([aaa, bbb])
#        print(LA.norm(aaaa - bbbb))
        scores, l2, _ = sess.run([output, l2dist_real, bbb])

        return scores, l2

    return doit

def evaluate_perturbation_kb_restore(args, sess, model, inputs):
    layernum = args['layer_number']
    batch_size = inputs.shape[0]
    aaa = model.model.layers[layernum].kernel
    aaa2 = model.model.layers[layernum].bias
    timg = tf.Variable(np.zeros(inputs.shape), dtype=tf.float32)
    tdelt_kernel = tf.Variable(np.zeros(aaa.shape, dtype=np.float32))
    tdelt_bias = tf.Variable(np.zeros(aaa2.shape, dtype=np.float32))
    tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

    assign_tdelt_kernel = tf.placeholder(tf.float32, aaa.shape)
    assign_tdelt_bias = tf.placeholder(tf.float32, aaa2.shape)
    assign_timg = tf.placeholder(tf.float32, inputs.shape)
    assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))

    model.model.layers[layernum].kernel = model.model.layers[layernum].kernel - tdelt_kernel
    model.model.layers[layernum].bias = model.model.layers[layernum].bias - tdelt_bias

    bbb = model.model.layers[layernum].kernel
    output = model.predict(timg)

   # ccc = model.model.layers[layernum].kernel
  #  ccc2 = model.model.layers[layernum].bias

    # these are the variables to initialize when we run
    setup = []
    setup.append(timg.assign(assign_timg))
    setup.append(tlab.assign(assign_tlab))
    setup.append(tdelt_kernel.assign(assign_tdelt_kernel))
    setup.append(tdelt_bias.assign(assign_tdelt_bias))

    def doit(inputs, targets, adv):
        batch = inputs[:batch_size]
        batchlab = targets[:batch_size]
        layernum = args['layer_number']
        akernel = model.model.layers[layernum].kernel
        abias = model.model.layers[layernum].bias
        z1 = adv[0: akernel.shape[0] * akernel.shape[1]]
        z2 = adv[akernel.shape[0] * akernel.shape[1]:]
        z1 = np.reshape(z1, akernel.shape)
        z2 = np.reshape(z2, abias.shape)
        sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tdelt_kernel: z1, assign_tdelt_bias: z2,})
        aaaa, bbbb = sess.run([aaa, bbb])
#        print(LA.norm(aaaa - bbbb))
        scores, _ = sess.run([output, aaa])

        return scores

    return doit

def evaluate_perturbation_testset(args, sess, model, inputs):

    batch_size = 10000
    timg = tf.Variable(np.zeros(inputs.shape), dtype=tf.float32)
    tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

    assign_timg = tf.placeholder(tf.float32, inputs.shape)
    assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))

    output = model.predict(timg)

    # these are the variables to initialize when we run
    setup = []
    setup.append(timg.assign(assign_timg))
    setup.append(tlab.assign(assign_tlab))

    def doit(inputs, targets):
        batch = inputs[:10000]
        batchlab = targets[:10000]

        sess.run(setup, {assign_timg: batch, assign_tlab: batchlab,})
#        print(LA.norm(aaaa - bbbb))
        scores, _ = sess.run([output, tlab])

        return scores

    return doit


def l1_l2_li_computation(args, data, model, adv, inception, inputs, targets, labels, true_ids):

    r_best = []
    d_best_l1 = []
    d_best_l2 = []
    d_best_linf = []
    r_average = []
    d_average_l1 = []
    d_average_l2 = []
    d_average_linf = []
    r_worst = []
    d_worst_l1 = []
    d_worst_l2 = []
    d_worst_linf = []

    if (args['show']):
        if not os.path.exists(str(args['save']) + "/" + str(args['dataset']) + "/" + str(args['attack'])):
            os.makedirs(str(args['save']) + "/" + str(args['dataset']) + "/" + str(args['attack']))

    for i in range(0, len(inputs), args['target_number']):
        pred = []
        for j in range(i, i + args['target_number']):
            if inception:
                pred.append(np.reshape(model.model.predict(adv[j:j + 1]), (data.test_labels[0:1].shape)))
            else:
                pred.append(model.model.predict(adv[j:j + 1]))

        dist_l1 = 1e10
        dist_l2 = 1e10
        dist_linf = 1e10
        dist_l1_index = 1e10
        dist_l2_index = 1e10
        dist_linf_index = 1e10
        for k, j in enumerate(range(i, i + args['target_number'])):
            if (np.argmax(pred[k], 1) == np.argmax(targets[j:j + 1], 1)):
                if (np.sum(np.abs(adv[j] - inputs[j])) < dist_l1):
                    dist_l1 = np.sum(np.abs(adv[j] - inputs[j]))
                    dist_l1_index = j
                if (np.amax(np.abs(adv[j] - inputs[j])) < dist_linf):
                    dist_linf = np.amax(np.abs(adv[j] - inputs[j]))
                    dist_linf_index = j
                if ((np.sum((adv[j] - inputs[j]) ** 2) ** .5) < dist_l2):
                    dist_l2 = (np.sum((adv[j] - inputs[j]) ** 2) ** .5)
                    dist_l2_index = j
        if (dist_l1_index != 1e10):
            d_best_l2.append((np.sum((adv[dist_l2_index] - inputs[dist_l2_index]) ** 2) ** .5))
            d_best_l1.append(np.sum(np.abs(adv[dist_l1_index] - inputs[dist_l1_index])))
            d_best_linf.append(np.amax(np.abs(adv[dist_linf_index] - inputs[dist_linf_index])))
            r_best.append(1)
        else:
            r_best.append(0)

        rand_int = np.random.randint(i, i + args['target_number'])
        if inception:
            pred_r = np.reshape(model.model.predict(adv[rand_int:rand_int + 1]), (data.test_labels[0:1].shape))
        else:
            pred_r = model.model.predict(adv[rand_int:rand_int + 1])
        if (np.argmax(pred_r, 1) == np.argmax(targets[rand_int:rand_int + 1], 1)):
            r_average.append(1)
            d_average_l2.append(np.sum((adv[rand_int] - inputs[rand_int]) ** 2) ** .5)
            d_average_l1.append(np.sum(np.abs(adv[rand_int] - inputs[rand_int])))
            d_average_linf.append(np.amax(np.abs(adv[rand_int] - inputs[rand_int])))

        else:
            r_average.append(0)

        dist_l1 = 0
        dist_l1_index = 1e10
        dist_linf = 0
        dist_linf_index = 1e10
        dist_l2 = 0
        dist_l2_index = 1e10
        for k, j in enumerate(range(i, i + args['target_number'])):
            if (np.argmax(pred[k], 1) != np.argmax(targets[j:j + 1], 1)):
                r_worst.append(0)
                dist_l1_index = 1e10
                dist_l2_index = 1e10
                dist_linf_index = 1e10
                break
            else:
                if (np.sum(np.abs(adv[j] - inputs[j])) > dist_l1):
                    dist_l1 = np.sum(np.abs(adv[j] - inputs[j]))
                    dist_l1_index = j
                if (np.amax(np.abs(adv[j] - inputs[j])) > dist_linf):
                    dist_linf = np.amax(np.abs(adv[j] - inputs[j]))
                    dist_linf_index = j
                if ((np.sum((adv[j] - inputs[j]) ** 2) ** .5) > dist_l2):
                    dist_l2 = (np.sum((adv[j] - inputs[j]) ** 2) ** .5)
                    dist_l2_index = j
        if (dist_l1_index != 1e10):
            d_worst_l2.append((np.sum((adv[dist_l2_index] - inputs[dist_l2_index]) ** 2) ** .5))
            d_worst_l1.append(np.sum(np.abs(adv[dist_l1_index] - inputs[dist_l1_index])))
            d_worst_linf.append(np.amax(np.abs(adv[dist_linf_index] - inputs[dist_linf_index])))
            r_worst.append(1)

        if (args['show']):
            for j in range(i, i + args['batch_size']):
                target_id = np.argmax(targets[j:j + 1], 1)
                label_id = np.argmax(labels[j:j + 1], 1)
                prev_id = np.argmax(np.reshape(model.model.predict(inputs[j:j + 1]),
                                               (data.test_labels[0:1].shape)), 1)
                adv_id = np.argmax(np.reshape(model.model.predict(adv[j:j + 1]), (data.test_labels[0:1].shape)), 1)
                suffix = "id{}_seq{}_lbl{}_prev{}_adv{}_{}_l1_{:.3f}_l2_{:.3f}_linf_{:.3f}".format(
                    true_ids[i],
                    target_id,
                    label_id,
                    prev_id,
                    adv_id,
                    adv_id == target_id,
                    np.sum(np.abs(adv[j] - inputs[j])),
                    np.sum((adv[j] - inputs[j]) ** 2) ** .5,
                    np.amax(np.abs(adv[j] - inputs[j])))

                show(inputs[j:j + 1], str(args['save']) + "/" + str(args['dataset']) + "/" + str(
                    args['attack']) + "/original_{}.png".format(suffix))
                show(adv[j:j + 1], str(args['save']) + "/" + str(args['dataset']) + "/" + str(
                    args['attack']) + "/adversarial_{}.png".format(suffix))

    print('best_case_L1_mean', np.mean(d_best_l1))
    print('best_case_L2_mean', np.mean(d_best_l2))
    print('best_case_Linf_mean', np.mean(d_best_linf))
    print('best_case_prob', np.mean(r_best))
    print('average_case_L1_mean', np.mean(d_average_l1))
    print('average_case_L2_mean', np.mean(d_average_l2))
    print('average_case_Linf_mean', np.mean(d_average_linf))
    print('average_case_prob', np.mean(r_average))
    print('worst_case_L1_mean', np.mean(d_worst_l1))
    print('worst_case_L2_mean', np.mean(d_worst_l2))
    print('worst_case_Linf_mean', np.mean(d_worst_linf))
    print('worst_case_prob', np.mean(r_worst))


def l0_computation(args, data, model, adv, inception, inputs, targets, labels, true_ids):
    r_best = []
    d_best_l1 = []
    r_average = []
    d_average_l1 = []
    r_worst = []
    d_worst_l1 = []

    if args['show']:
        if not os.path.exists(str(args['save']) + "/" + str(args['dataset']) + "/" + str(args['attack'])):
            os.makedirs(str(args['save']) + "/" + str(args['dataset']) + "/" + str(args['attack']))

    for i in range(0, len(inputs), args['target_number']):
        pred = []
        for j in range(i, i + args['target_number']):
            if inception:
                pred.append(np.reshape(model.model.predict(adv[j:j + 1]), (data.test_labels[0:1].shape)))
            else:
                pred.append(model.model.predict(adv[j:j + 1]))

        dist_l1 = 1e10
        dist_l1_index = 1e10

        for k, j in enumerate(range(i, i + args['target_number'])):
            if np.argmax(pred[k], 1) == np.argmax(targets[j:j + 1], 1):
                #if np.array(np.nonzero(adv[j]-inputs[j])).shape[1] < dist_l1:
                if np.array(np.nonzero(np.where(np.abs(adv[j]-inputs[j]) < 1e-7, 0, adv[j]-inputs[j]))).shape[1] < dist_l1:
                    dist_l1 = np.array(np.nonzero(np.where(np.abs(adv[j]-inputs[j]) < 1e-7, 0, adv[j]-inputs[j]))).shape[1]
                    #abc = np.array(adv[j]-inputs[j])
                    #print(np.nonzero(np.where(adv[j] - inputs[j] < 1e-8, 0, adv[j] - inputs[j])))
                    dist_l1_index = j
        if dist_l1_index != 1e10:
            d_best_l1.append(np.array(np.nonzero(np.where(np.abs(adv[dist_l1_index]-inputs[dist_l1_index]) < 1e-7, 0,
                                                          adv[dist_l1_index]-inputs[dist_l1_index]))).shape[1])
            r_best.append(1)
        else:
            r_best.append(0)

        rand_int = np.random.randint(i, i + args['target_number'])
        if inception:
            pred_r = np.reshape(model.model.predict(adv[rand_int:rand_int + 1]), (data.test_labels[0:1].shape))
        else:
            pred_r = model.model.predict(adv[rand_int:rand_int + 1])
        if np.argmax(pred_r, 1) == np.argmax(targets[rand_int:rand_int + 1], 1):
            r_average.append(1)
            d_average_l1.append(np.array(np.nonzero(np.where(np.abs(adv[rand_int]-inputs[rand_int]) < 1e-7, 0,
                                                             adv[rand_int]-inputs[rand_int]))).shape[1])
        else:
            r_average.append(0)

        dist_l1 = 0
        dist_l1_index = 1e10
        for k, j in enumerate(range(i, i + args['target_number'])):
            if (np.argmax(pred[k], 1) != np.argmax(targets[j:j + 1], 1)):
                r_worst.append(0)
                dist_l1_index = 1e10
                break
            else:
                if np.array(np.nonzero(np.where(np.abs(adv[j]-inputs[j]) < 1e-7, 0, adv[j]-inputs[j]))).shape[1] > dist_l1:
                    dist_l1 = np.array(np.nonzero(np.where(np.abs(adv[j]-inputs[j]) < 1e-7, 0, adv[j]-inputs[j]))).shape[1]
                    dist_l1_index = j

        if dist_l1_index != 1e10:
            d_worst_l1.append(np.array(np.nonzero(np.where(np.abs(adv[dist_l1_index]-inputs[dist_l1_index]) < 1e-7, 0,
                                                           adv[dist_l1_index]-inputs[dist_l1_index]))).shape[1])
            r_worst.append(1)

        if args['show']:
            for j in range(i, i + args['batch_size']):
                target_id = np.argmax(targets[j:j + 1], 1)
                label_id = np.argmax(labels[j:j + 1], 1)
                prev_id = np.argmax(np.reshape(model.model.predict(inputs[j:j + 1]),
                                               (data.test_labels[0:1].shape)), 1)
                adv_id = np.argmax(np.reshape(model.model.predict(adv[j:j + 1]), (data.test_labels[0:1].shape)), 1)
                suffix = "id{}_seq{}_lbl{}_prev{}_adv{}_{}_l1_{:.3f}_l2_{:.3f}_linf_{:.3f}".format(
                    true_ids[i],
                    target_id,
                    label_id,
                    prev_id,
                    adv_id,
                    adv_id == target_id,
                    np.sum(np.abs(adv[j] - inputs[j])),
                    np.sum((adv[j] - inputs[j]) ** 2) ** .5,
                    np.amax(np.abs(adv[j] - inputs[j])))

                show(inputs[j:j + 1], str(args['save']) + "/" + str(args['dataset']) + "/" + str(
                    args['attack']) + "/original_{}.png".format(suffix))
                show(adv[j:j + 1], str(args['save']) + "/" + str(args['dataset']) + "/" + str(
                    args['attack']) + "/adversarial_{}.png".format(suffix))

    print('best_case_L0_mean', np.mean(d_best_l1))
    print('best_case_prob', np.mean(r_best))
    print('average_case_L0_mean', np.mean(d_average_l1))
    print('average_case_prob', np.mean(r_average))
    print('worst_case_L0_mean', np.mean(d_worst_l1))
    print('worst_case_prob', np.mean(r_worst))

def main(args):

    with tf.Session() as sess:
        if args['dataset'] == 'mnist':
            data, model = MNIST(), MNISTModel("models/mnist", sess)
            handpick = False
            inception = False
        if args['dataset'] == "cifar":
            data, model = CIFAR(), CIFARModel("models/cifar", sess)
            handpick = True
            inception = False
        if args['dataset'] == "imagenet":
            data, model = ImageNet(args['seed_imagenet']), InceptionModel(sess)
            handpick = True
            inception = True

        if args['adversarial'] != "none":
            model = MNISTModel("models/mnist_cwl2_admm" + str(args['adversarial']), sess)

        if args['temp'] and args['dataset'] == 'mnist':
            model = MNISTModel("models/mnist-distilled-" + str(args['temp']), sess)
        if args['temp'] and args['dataset'] == 'cifar':
            model = CIFARModel("models/cifar-distilled-" + str(args['temp']), sess)

        inputs, targets, labels, true_ids = generate_data_ST(data, model, samples=args['numimg'],
                                                             samplesT=args['numimgT'], targeted=True,
                                        start=0, inception=inception, handpick=handpick, seed=args['seed'])
        #print(true_ids)
        if args['attack'] == 'L2C':
            attack = CarliniL2(sess, model, batch_size=args['batch_size'], max_iterations=args['maxiter'],
                               confidence=args['conf'],
                               binary_search_steps=args['binary_steps'],
                               abort_early=args['abort_early'])

        if args['attack'] == 'L2LA2':
            attack = LADMML2re(sess, model, batch_size=args['batch_size'], max_iterations=args['maxiter'],
                               layernum=args['layer_number'], use_kernel=args['use_kernel'],
                               confidence=args['conf'], binary_search_steps=args['iteration_steps'], ro=args['ro'],
                               abort_early=args['abort_early'])


        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.\n")

        if args['conf'] != 0:
            model = MNISTModel("models/mnist-distilled-100", sess)

        if args['kernel_bias']:
            EP = evaluate_perturbation_kb(args, sess, model, inputs)
            scores, l2 = EP(inputs, targets, adv)
            EPT = evaluate_perturbation_testset(args, sess, model, data.test_data)
            test_scores = EPT(data.test_data, data.test_labels)
            EP2 = evaluate_perturbation_kb_restore(args, sess, model, inputs)
            scores2 = EP2(inputs, targets, adv)
            EPT2 = evaluate_perturbation_testset(args, sess, model, data.test_data)
            test_scores2 = EPT2(data.test_data, data.test_labels)
        else:
            EP = evaluate_perturbation(args, sess, model, inputs)
#        scores = EP(inputs, targets, adv)
#        scores2 = EP2(inputs, targets, adv)

        score_count = []
        score_count2 = []
        score_count3 = []

        score_count4 = []
        for e, (sc) in enumerate(scores):

            if np.argmax(sc) == np.argmax(targets[e]):
                score_count.append(1)
                if e < args['numimg']:
                    score_count4.append(1)
            else:
                score_count.append(0)
                if e < args['numimg']:
                    score_count4.append(0)

        for e, (sc) in enumerate(scores):
            if np.argmax(sc) == np.argmax(labels[e]):
                score_count3.append(1)
            else:
                score_count3.append(0)

        for e, (sc2) in enumerate(scores2):
            if np.argmax(sc2) == np.argmax(labels[e]):
                score_count2.append(1)
            else:
                score_count2.append(0)

        test_score_count = []
        test_score_count2 = []

        for e, (tsc) in enumerate(test_scores):

            if np.argmax(tsc) == np.argmax(data.test_labels[e]):
                test_score_count.append(1)
            else:
                test_score_count.append(0)

        for e, (tsc2) in enumerate(test_scores2):

            if np.argmax(tsc2) == np.argmax(data.test_labels[e]):
                test_score_count2.append(1)
            else:
                test_score_count2.append(0)

        l0s = np.count_nonzero(adv)
        successrate = np.mean(score_count)
        successrate2 = np.mean(score_count2)
        successrate3 = np.mean(score_count3)
        test_successrate = np.mean(test_score_count)
        test_successrate2 = np.mean(test_score_count2)

        print('original model, success rate of T images for the original labels:', successrate2)
        print('modified model, success rate of T images for the original labels:', successrate3)
        print('modified model, success rate of T images for the target labels:', successrate)
        print('modified model, success rate of S imges for the target labels:', np.mean(score_count4))

        print('modified model, success rate of test set for the original labels:', test_successrate)
        print('original model, success rate of test set for the original labels:', test_successrate2)
        print('l0 distance:', l0s)
        print('l2 distance:', l2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar", "imagenet"], default="mnist",
                        help="dataset to use")
    parser.add_argument("-nS", "--numimg", type=int, default=4, help="number of images to attack")
    parser.add_argument("-nT", "--numimgT", type=int, default=2000, help="number of images to attack")
    parser.add_argument("-ln", "--layer_number", type=int, default=15, help="the layer to perform perturbations")
    parser.add_argument("-kb", "--use_kernel", action='store_true', default=True,
                        help="use kernel or bias")
    parser.add_argument("-bkb", "--kernel_bias", action='store_true', default=True,
                        help="use kernel and bias")
    parser.add_argument("-b", "--batch_size", type=int, default=2000, help="batch size")
    parser.add_argument("-m", "--maxiter", type=int, default=1000, help="max iterations per bss")
    parser.add_argument("-is", "--iteration_steps", type=int, default=200, help="number of iteration")
    parser.add_argument("-ro", "--ro", type=float, default=100, help="value of ro")
    parser.add_argument("-bs", "--binary_steps", type=int, default=9, help="number of bss")
    parser.add_argument("-ae", "--abort_early", action='store_true', default=False,
                        help="abort binary search step early when losses stop decreasing")
    parser.add_argument("-cf", "--conf", type=int, default=0, help='Set attack confidence for transferability tests')
    parser.add_argument("-imgsd", "--seed_imagenet", type=int, default=1,
                        help='random seed for pulling images from ImageNet test set')
    parser.add_argument("-sd", "--seed", type=int, default=111,
                        help='random seed for pulling images from data set')
    parser.add_argument("-sh", "--show", action='store_true', default=False,
                        help='save original and adversarial images to save directory')
    parser.add_argument("-s", "--save", default="./saves", help="save directory")
    parser.add_argument("-a", "--attack",
                        choices=["L2C", "L2A", "L2AE", "L2LA", "L2LAE", "L2LAE2"],
                        default="L2LA2",
                        help="attack algorithm")
    parser.add_argument("-tn", "--target_number", type=int, default=9, help="number of targets for one input")
    parser.add_argument("-tr", "--train", action='store_true', default=False,
                        help="save adversarial images generated from train set")
    parser.add_argument("-tp", "--temp", type=int, default=0,
                        help="attack defensively distilled network trained with this temperature")
    parser.add_argument("-adv", "--adversarial", choices=["none", "l2", "l1", "en", "l2l1", "l2en"], default="none",
                        help="attack network adversarially trained under these examples")
    parser.add_argument("-be", "--beta", type=float, default=1e-2, help='beta hyperparameter')
    args = vars(parser.parse_args())
    print(args)
    main(args)
