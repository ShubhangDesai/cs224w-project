import torch
import torch.nn.functional as F
import numpy as np

def flag_train(model, data, train_label, train_idx, loss_fn, dropedge_rate, flag_steps, flag_step_size):
	print("doing_flag")

	# Initialize adversarial perturbations
	perturb = torch.FloatTensor(data.x.shape).uniform_(-flag_step_size, flag_step_size) # Uniformation perturbations
	perturb = perturb.to('cuda' if torch.cuda.is_available() else 'cpu').requires_grad_()

	out = model(data.x + perturb, data.adj_t)[train_idx]
	loss = loss_fn(out, train_label) / flag_steps

	for _ in range(flag_steps): # Gradient ascent
		loss.backward()
		perturb.data = (perturb.detach() + flag_step_size*torch.sign(perturb.grad.detach())).data # Perturbation gradient ascent
		perturb.grad[:] = 0.

		out = model(data.x + perturb, data.adj_t)[train_idx]
		loss = loss_fn(out, train_label) / flag_steps

	return loss, out
