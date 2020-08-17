## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
from numpy import linalg as LA

BINARY_SEARCH_STEPS = 30  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
RO = 20
LAYERNUMBER=15
USEKERNEL=True
KERNELBIAS=True
SS = 10

class LADMML2re:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE, layernum=LAYERNUMBER,
                 targeted=TARGETED, learning_rate=LEARNING_RATE, s=SS,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, ro=RO, use_kernel=USEKERNEL, kernel_bias=KERNELBIAS):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.use_kernel = use_kernel
        self.ro = ro
        self.s = s
        self.layernum = layernum
        self.kernel_bias = kernel_bias
        self.grad = self.gradient_descent(sess, model)

    def compare(self, x, y):
        if not isinstance(x, (float, int, np.int64)):
            x = np.copy(x)
            if self.TARGETED:
                x[y] -= self.CONFIDENCE
            else:
                x[y] += self.CONFIDENCE
            x = np.argmax(x)
        if self.TARGETED:
            return x == y
        else:
            return x != y

    def gradient_descent(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)
        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))

        if not self.kernel_bias:

            if self.use_kernel:
                aaa = model.model.layers[self.layernum].kernel
            else:
                aaa = model.model.layers[self.layernum].bias

            tdelt = tf.Variable(np.zeros(aaa.shape, dtype=np.float32))
            assign_tdelt = tf.placeholder(tf.float32, aaa.shape)

            if self.use_kernel:
                model.model.layers[self.layernum].kernel = tdelt + model.model.layers[self.layernum].kernel
                bbb = model.model.layers[self.layernum].kernel
            else:
                model.model.layers[self.layernum].bias = tdelt + model.model.layers[self.layernum].bias
                bbb = model.model.layers[self.layernum].bias

        else:
            aaa = model.model.layers[self.layernum].kernel
            aaa2 = model.model.layers[self.layernum].bias

            tdelt_kernel = tf.Variable(np.zeros(aaa.shape, dtype=np.float32))
            assign_tdelt_kernel = tf.placeholder(tf.float32, aaa.shape)
            tdelt_bias = tf.Variable(np.zeros(aaa2.shape, dtype=np.float32))
            assign_tdelt_bias = tf.placeholder(tf.float32, aaa2.shape)

            model.model.layers[self.layernum].kernel = tdelt_kernel + model.model.layers[self.layernum].kernel
            model.model.layers[self.layernum].bias = tdelt_bias + model.model.layers[self.layernum].bias
            bbb = model.model.layers[self.layernum].kernel
            bbb2 = model.model.layers[self.layernum].bias

        output = model.predict(timg)
        l2dist_real = tf.reduce_sum(tf.square(tdelt_kernel)) + tf.reduce_sum(tf.square(tdelt_bias))
        l2dist_real = tf.sqrt(l2dist_real)
        real = tf.reduce_sum(tlab * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        wei = np.ones(batch_size)
        wei[0:self.s] = 1000.0 * wei[0:self.s]
        loss1 = loss1 * wei
        loss1 = 0.5 * tf.reduce_sum(loss1)

        grad_tdelt_kernel, grad_tdelt_bias = tf.gradients(loss1, [tdelt_kernel, tdelt_bias])

    #    model.model.layers[13].kernel = model.model.layers[13].kernel - tdelt
        if not self.kernel_bias:
            if self.use_kernel:
                model.model.layers[self.layernum].kernel = aaa
                ccc = model.model.layers[self.layernum].kernel
            else:
                model.model.layers[self.layernum].bias = aaa
                ccc = model.model.layers[self.layernum].bias
        else:
            model.model.layers[self.layernum].kernel = aaa
            model.model.layers[self.layernum].bias = aaa2

            ccc = model.model.layers[self.layernum].kernel
            ccc2 = model.model.layers[self.layernum].bias

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(tdelt_kernel.assign(assign_tdelt_kernel))
        setup.append(tdelt_bias.assign(assign_tdelt_bias))

        def doit(imgs, labs, z):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
            akernel = model.model.layers[self.layernum].kernel
            abias = model.model.layers[self.layernum].bias
            z1 = z[0: akernel.shape[0] * akernel.shape[1]]
            z2 = z[akernel.shape[0] * akernel.shape[1]:]
            z1 = np.reshape(z1, akernel.shape)
            z2 = np.reshape(z2, abias.shape)

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tdelt_kernel: z1, assign_tdelt_bias:z2})
            aaaa, bbbb, cccc = sess.run([aaa, bbb, ccc])
      #      print(LA.norm(aaaa - bbbb))
      #      print(LA.norm(aaaa - cccc))
            scores, l2dist, delt_grad_kernel, delt_grad_bias = sess.run([output, l2dist_real,
                                                                         grad_tdelt_kernel, grad_tdelt_bias])

            delt_gradss = np.hstack((np.reshape(delt_grad_kernel, (-1)), np.reshape(delt_grad_bias, (-1))))
            return scores, l2dist, np.array(delt_gradss)

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r.extend(self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size
        if self.kernel_bias:
            aab = self.model.model.layers[self.layernum].kernel
            aab2 = self.model.model.layers[self.layernum].bias

        o_bestl2 = 1e10
        o_bestattack = 0.0 * np.ones(aab.shape[0]*aab.shape[1] + aab2.shape[0])
        o_successrate = 0.0

        delt = 0.0 * np.ones(aab.shape[0]*aab.shape[1] + aab2.shape[0])
        s = 0.0 * np.ones(aab.shape[0]*aab.shape[1] + aab2.shape[0])

        alpha = 20
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_step, o_bestl2)

            temp = delt - s
#            z = np.where(np.abs(temp) ** 2 < (2.0 / self.ro), 0, temp)
            z = self.ro/(2.0 + self.ro) * temp

            scor, _, delt_grads = self.grad(imgs, labs, delt)

            eta = 1/np.sqrt(outer_step+1)
            delt = 1/(alpha / eta * imgs.shape[0] +  self.ro) * \
                ( self.ro * (z + s) + alpha / eta * imgs.shape[0] * delt - delt_grads)

            scores, l2, _ = self.grad(imgs, labs, delt)
            s = s + z - delt

            score_count = []
            for e, (sc) in enumerate(scores):
 #               if e < self.s:
                if self.compare(sc, np.argmax(labs[e])):
                    score_count.append(1)
                else:
                    score_count.append(0)

            successrate = np.mean(score_count)

            print(successrate)
            print(l2)
            if successrate >= o_successrate:
                o_successrate = successrate
                l0s = np.count_nonzero(delt)
                o_bestl2 = l0s
                o_bestattack = delt
                scores_backup = scores

        return o_bestattack
