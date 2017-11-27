
"""
 This is the equation:
     y = 3x1 + 5x2 + 6x3

     We are going to find the coefficients of this equation through gradient
     descent. We are going to use squared error as loss function.
"""

import random
import math

class LogisticGradDesSol():
    """This class finds the solution of an equation through gradient descent"""

    def __init__(self, learning_rate):
        self.w1 = 0.0
        self.w2 = 0.0
        self.w3 = 0.0
        self.alpha = learning_rate
        self.clear_loss_grad()

    def find_actual_value(self, x1, x2, x3):
        yy = 3*x1 + 5*x2 + 6*x3
        if yy > 0:
            return 1
        else:
            return 0
 
    def find_estimated_value(self, x1, x2, x3):
        hx = self.w1*x1 + self.w2*x2 + self.w3*x3
        return 1/(1+math.exp(-1*hx))

    def clear_loss_grad(self):
        self.total_loss = 0.0
        self.loss_wrt_w1 = 0.0
        self.loss_wrt_w2 = 0.0
        self.loss_wrt_w3 = 0.0

    def print_learnt_params(self):
        print ("w1: %.2f" % self.w1, " ; w2: %.2f" % self.w2, " ; w3: %.2f" % self.w3, "; total loss: %.5f" % self.total_loss)

    def execute_single_batch_gd(self, num_ex):
        if num_ex <= 0:
            return
            
        self.clear_loss_grad()
        for i in range(num_ex):
            rand_num1 = random.uniform(0.0, 10.0)
            rand_num2 = random.uniform(10.0, 20.0)
            rand_num3 = random.uniform(0.0, 5.0)
            y = self.find_actual_value(rand_num1, rand_num2, rand_num3)
            hx = self.find_estimated_value(rand_num1, rand_num2, rand_num3)
            delta_estimation = hx - y
            self.total_loss += -1*y*math.log(hx) - (1-y)*math.log(1-hx)
            self.loss_wrt_w1 += delta_estimation * rand_num1
            self.loss_wrt_w2 += delta_estimation * rand_num2
            self.loss_wrt_w3 += delta_estimation * rand_num3
        # update the variable
        self.total_loss /= num_ex
        self.w1 -= self.alpha * self.loss_wrt_w1 / num_ex
        self.w2 -= self.alpha * self.loss_wrt_w2 / num_ex
        self.w3 -= self.alpha * self.loss_wrt_w3 / num_ex
        self.print_learnt_params();

    def apply_grad_des(self, batch_size, max_itr = 1000):
        itr = 0;
        batch_converged = 0;
        while True:
            itr+=1
            self.execute_single_batch_gd(batch_size)
            if self.total_loss <= 0.0001:
                batch_converged +=1
            elif itr > max_itr:
                print("not converged event after ", max_itr, " iterations")
                break;
            else:
                batch_converged = 0

            if batch_converged > 5:
                print("gradient descent converged")
                break


print ("hello_world")
grad_des = LogisticGradDesSol(0.05)
grad_des.apply_grad_des(5, 500)








