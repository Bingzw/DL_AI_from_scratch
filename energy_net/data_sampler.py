import torch
import random
import numpy as np


class Sampler:
    def __init__(self, model, img_shape, sample_size, max_len=8192):
        """
        :param model: Neural network model to use for modeling E_theta
        :param img_shape: shape of the images to model
        :param sample_size: batch size of the samples
        :param max_len: maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        # each sampled element shape is (1, img_shape), converted the sample values to 0~1
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_examples(self, steps=60, step_size=10, device="cuda"):
        """
        Functions for getting a new batch of "fake" images
        :param steps: Number of iterations in the MCMC algorithm
        :param step_size: Learning rate for the MCMC using Langevin dynamics
        :return: New batch of images
        """
        # choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape)*2-1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([old_imgs, rand_imgs], dim=0).detach().to(device)

        # perform MCMC sampling
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

        # add new images to the buffer and remove old ones if necessary
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        """
        Function for generating samples using Langevin dynamics
        :param model: Neural network model to use for modeling E_theta
        :param inp_imgs: Input images to start the sampling
        :param steps: Number of iterations in the MCMC algorithm
        :param step_size: Learning rate for the MCMC using Langevin dynamics
        :param return_img_per_step: If True, return images at each step
        :return: Images generated using Langevin dynamics
        """
        # before MCMC, set the model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)  # simply get random number with the shape of the input images

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over the number of steps
        for _ in range(steps):
            # Add noise to the input images
            noise.normal_(0, 0.005)  # sample noise from a normal distribution in place
            inp_imgs.data.add_(noise.data)  # calling add_ on the data attribute of the tensor does not affect gradient tracking
            inp_imgs.data.clamp(min=-1.0, max=1.0)  # cap the values to -1~1
            # Calculate gradients for the current input
            out_imgs = -model(inp_imgs) # the model represents -E_theta
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stability, we clip the gradients
            # Apply gradients to current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()  # gradient calculated in the current iteration does not affect subsequent iterations
            inp_imgs.grad.zero_()  # avoid accumulating gradients from previous iterations due to line 84
            inp_imgs.data.clamp_(-1.0, 1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.detach().clone())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
