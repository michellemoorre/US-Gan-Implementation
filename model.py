import torch
import torch.nn.functional as F

class InputExtractor(torch.nn.Module):
	@property
	def device(self):
		return next(self.parameters()).device
	
	def forward(self, x, c):
		c = c.view(c.size(0), c.size(1), 1, 1)
		c = c.repeat(1, 1, x.size(2), x.size(3))
		x = torch.cat([x,c], dim=1)
		return x

def ConvBlck(in_chanels, out_chanels, *arg, **kwargs):
	return torch.nn.Sequential(
		torch.nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=2, padding=1, bias=False),
		torch.nn.InstanceNorm2d(conv_dim, affine=True),
		torch.nn.LeakyReLU(negative_slope=0.2)
	)

def ConvTBlck(in_chanels, out_chanels, *arg, **kwargs):
	return torch.nn.Sequential(
		torch.nn.ConvTranspose2d(in_chanels, out_chanels, kernel_size=3, stride=2, padding=1, output_padding=1),
		torch.nn.InstanceNorm2d(conv_dim, affine=True),
		torch.nn.LeakyReLU(negative_slope=0.2)
	)

class ResidualBlck(torch.nn.Module):
	@property
	def device(self):
		return next(self.parameters()).device

	def __init__(self, in_chanels, out_chanels):
		self.blcks = torch.nn.Sequential(
			nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(out_chanels, affine=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_chanels, out_chanels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(out_chanels, affine=True)
		)
	
	def forward(self, x):
		return x + self.blcks(x)

class USGenerator(torch.nn.Module):
	@property
	def device(self):
		return next(self.parameters()).device

	def __init__(self, C):
		self.C = C
		
		self.blcks = torch.nn.Sequential(
			InputExtractor(),
			ConvBlck(in_chanels=3+C, out_chanels=64),      #in: DxDx(3+C) --> DxDx64 :out
			ConvBlck(in_chanels=64, out_chanels=128),      #in: DxDx64 --> D/2xD/2x128 :out
			ConvBlck(in_chanels=128, out_chanels=256),     #in: D/2xD/2x128 --> D/4xD/4x256 :out
			ResidualBlck(in_chanels=256, out_chanels=256), #in: D/4xD/4x256 === D/4xD/4x256 :out
			ConvTBlck(in_chanels=256, out_chanels=128),    #in: D/4xD/4x256 --> D/2xD/2x128 :out
			ConvTBlck(in_chanels=128, out_chanels=64),     #in: D/2xD/2x128 --> DxDx64 :out
			ConvTBlck(in_chanels=64, out_chanels=3)        #in: DxDx64 --> DxDx3 :out
		)
	
	def forward(self, x, c):
		x += self.blcks(x)
		return x

class USDiscriminator(torch.nn.Module):
	@property
	def device(self):
		return next(self.parameters()).device

	def __init__(self, image_size=128, conv_dim=64, c_dim=5, blck_count=6):
		super().__init__()

		layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, blck_count):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
		
		kernel_size = int(image_size / np.power(2, repeat_num))
		self.main = torch.nn.Sequential(*layers)
		self.prob_conv = torch.nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
		self.c_conv = torch.nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
	
	def forward(self, x):
		middle = self.main(x)
		c_pred = self.c_conv(x)
		prob_pred = self.prob_conv(x)
		
		return prob_pred, c_pred

class GAN(torch.nn.Module):
	@property
	def device(self):
		return next(self.parameters()).device

	def __init__(self):
		super().__init__()
		
		self.discriminator = USDiscriminator()
		self.generator = USGenerator()

	def adverserial_loss(self, X_real, X_fake):
		adverserial_loss_real = torch.mean(self.discriminator(X_real)[0])
		adverserial_loss_fake = torch.mean(self.discriminator(X_fake)[0])

		alpha = torch.rand(X_real.size(0), 1, 1, 1).to(self.device)
		X_hat = (alpha * X_real.data + (1 - alpha) * X_fake.data).requires_grad_(True)
		y_hat, _ = self.discriminator(X_hat)

		weight = torch.ones(y_hat.size()).to(self.device)
		gradient = torch.autograd.grad(outputs=y_hat,
								   inputs=X_hat,
								   grad_outputs=weight,
								   retain_graph=True,
								   create_graph=True,
								   only_inputs=True)[0]

		gradient = gradient.view(dydx.size(0), -1)
		gradient_norm = torch.sqrt(torch.sum(gradient**2, dim=1))
		grad_penalty = torch.mean((gradient_norm-1)**2)

		return adverserial_loss_real - adverserial_loss_fake - self.l_gp * grad_penalty

	def reconstruction_loss(self, X, c, c_original):

		return torch.mean(torch.abs(X - self.generator(self.generator(X, c), c_original)))

	def classification_loss(self, X, target):
		c_pred, _ = self.discriminator(X)
		return F.cross_entropy(c_pred, target)

	def generator_loss(self):

		return adverserial_loss() + self.l_c * classification_loss() + self.l_r * reconstruction_loss()

	def discriminator_loss(self):
		
		return -adverserial_loss() + self.l_c * classification_loss()