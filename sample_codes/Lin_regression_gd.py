
"""
 This is the equation:
     y = 3 + 3x1 + 4x2 + 100x3

     We are going to find the coefficients of this equation through gradient
     descent. We are going to use squared error as loss function.

     Two issues I saw initially in Lin Regression:
     1 .Somehow in this program, the bias term is not converging properly.
     Having a higher learning rate on bias term helps in converging it
     Similar discussion: https://goo.gl/9fqydH
     Without a bias term in actual function, things converge really well.

     2. If feature values are changed (say x3 is in range of 15 and 35), gd never
     converges. The loss becomes big and big over iterations and it overflows. Feature normalization
     helped in this case.
     Another way to fix is to use adaptive learning rate. RMSProp algorithm was best here.
     RMSProp increases or decreases alpha based on the loss. Depending on the type of gradient, it adjusts the
     alpha beautifully. However, if feature values are extremely large, then even RMSProp overflows. I am not sure
     how to solve that!!

     Also, instead of generating uniform rand number, generating
     normally distributed random numbers helped.

     Finally, generating positive as well as negative numbers helped a lot. So,
     in gradient decent, your inputs should be zero centered. By generating only
     positive normally distributed numbers, it never converged. While using positive
     as well as negative, it converged beautifully.

     Initializating the weights also helped a lot. The convergence happend a lot
     faster. The whole gradient decent is converging so beautifully including
     bias term.
"""

import random

class GradDesSol():
    """This class finds the solution of an equation through gradient descent"""

    def __init__(self, learning_rate,
                 use_rms_prop = False):
        # initializing the weights helped a lot
        self.w0 = random.normalvariate(0.0, 0.1)
        self.w1 = random.normalvariate(0.0, 0.1)
        self.w2 = random.normalvariate(0.0, 0.1)
        self.w3 = random.normalvariate(0.0, 0.1)
        self.alpha = learning_rate
        self.gamma = 0.5

        # for RMSProp algorithm
        self.use_rms_prop = use_rms_prop
        self.rms_grad_wrt_w0 = 0.0
        self.rms_grad_wrt_w1 = 0.0
        self.rms_grad_wrt_w2= 0.0
        self.rms_grad_wrt_w3 = 0.0        
        self.clear_loss_grad()

    def find_actual_value(self, x1, x2, x3):
        return 3 + 3*x1 + 4*x2 + 100*x3

    def find_estimated_value(self, x1, x2, x3):
        return self.w0 + self.w1*x1 + self.w2*x2 + self.w3*x3

    def clear_loss_grad(self):
        self.total_loss = 0.0
        self.loss_wrt_w0 = 0.0
        self.loss_wrt_w1 = 0.0
        self.loss_wrt_w2 = 0.0
        self.loss_wrt_w3 = 0.0

    def print_learnt_params(self):
        print ("w0: %.5f" % self.w0,
               "w1: %.5f" % self.w1,
               " ; w2: %.5f" % self.w2,
               " ; w3: %.5f" % self.w3,
               " ; Loss: %.3f" % self.total_loss
               )

    def execute_single_batch_gd(self, num_ex):
        if num_ex <= 0:
            return

        self.clear_loss_grad()
        for i in range(num_ex):
#           uniform generated numbers are poorer takes forever to converge
#            rand_num1 = random.uniform(-2.0, 1.0)
#            rand_num2 = random.uniform(-1.0, 2.0)
#            rand_num3 = random.uniform(-2.5, 1.5)

#           positive and negative normally distributed numbers converged the
#           regression quickly
            rand_num1 = random.normalvariate(10.0, 10.0)
            rand_num2 = random.normalvariate(3.0, 10.0)
            rand_num3 = random.normalvariate(5.0, 10.0)
            delta_estimation = (self.find_estimated_value(rand_num1, rand_num2, rand_num3) -
                                self.find_actual_value(rand_num1, rand_num2, rand_num3))
            self.total_loss += delta_estimation**2
            self.loss_wrt_w0 += delta_estimation
            self.loss_wrt_w1 += delta_estimation * rand_num1
            self.loss_wrt_w2 += delta_estimation * rand_num2
            self.loss_wrt_w3 += delta_estimation * rand_num3


        # compute average gradient per coordinate
        self.total_loss /= 2*num_ex
        self.loss_wrt_w0 /= num_ex
        self.loss_wrt_w1 /= num_ex
        self.loss_wrt_w2 /= num_ex
        self.loss_wrt_w3 /= num_ex

        # update the variable
        # This is the normal update mechanism. The current equation converges in between
        # 300-500 iterations
        if self.use_rms_prop == False:
            self.w0 -= self.alpha * self.loss_wrt_w0
            self.w1 -= self.alpha * self.loss_wrt_w1
            self.w2 -= self.alpha * self.loss_wrt_w2
            self.w3 -= self.alpha * self.loss_wrt_w3

        # This is the rmsprop mechanism. It is really amazing compared to plain single alpha
        # It makes the alpha increase or decrease depending on the loss. So, now input feature values
        # doesn't need to be normalized.

        # find the new RMS of the gradient per coordinate
        elif self.use_rms_prop:
            self.rms_grad_wrt_w0 = self.gamma*self.rms_grad_wrt_w0 +    \
                (1-self.gamma)*(self.loss_wrt_w0**2)
            self.rms_grad_wrt_w1 = self.gamma*self.rms_grad_wrt_w1 +    \
                (1-self.gamma)*(self.loss_wrt_w1**2)
            self.rms_grad_wrt_w2 = self.gamma*self.rms_grad_wrt_w2 +    \
                (1-self.gamma)*(self.loss_wrt_w2**2)
            self.rms_grad_wrt_w3 = self.gamma*self.rms_grad_wrt_w3 +    \
                (1-self.gamma)*(self.loss_wrt_w3**2)
            dec_alpha_w0 = max(0.0005, 2.0 * (self.alpha/(self.rms_grad_wrt_w0 + 0.0001)**0.5))
            dec_alpha_w1 = max(0.0005, (self.alpha/(self.rms_grad_wrt_w1 + 0.0001)**0.5))
            dec_alpha_w2 = max(0.0005, (self.alpha/(self.rms_grad_wrt_w2 + 0.0001)**0.5))
            dec_alpha_w3 = max(0.0005, (self.alpha/(self.rms_grad_wrt_w3 + 0.0001)**0.5))
            
            self.w0 -= dec_alpha_w0 * self.loss_wrt_w0
            self.w1 -= dec_alpha_w1 * self.loss_wrt_w1
            self.w2 -= dec_alpha_w2 * self.loss_wrt_w2
            self.w3 -= dec_alpha_w3 * self.loss_wrt_w3
            # print ("alpha_0: %.5f" % dec_alpha_w0,
            #    "alpha_1: %.5f" % dec_alpha_w1,
            #    " ; alpha_2: %.5f" % dec_alpha_w2,
            #    " ; alpha_3: %.5f" % dec_alpha_w3
            #    )

        self.print_learnt_params();


    def apply_grad_des(self, batch_size, max_itr = 1000):
        itr = 0;
        batch_converged = 0;
        while True:
            itr+=1
            self.execute_single_batch_gd(batch_size)
            if self.total_loss <= 0.001:  # this is not the ideal termination condition too.
                                          # we can check the percentage update on the parameters to determine the
                                          # termination conditions.
                                          # https://stackoverflow.com/questions/14464612
                batch_converged +=1
            elif itr > max_itr:
                print("not converged event after ", max_itr, " iterations")
                break;
            else:
                batch_converged = 0

            if batch_converged > 5:
                print("gradient descent converged in " , itr , " iterations.")
                break


print ("hello_world here")
grad_des = GradDesSol(0.005, False)     # using constant alpha alogrithm makes the model overflow. RMSProp helped here
grad_des.apply_grad_des(100, 50000)
