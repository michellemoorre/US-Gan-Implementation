import torch

class InputExtractor(torch.nn.Module):
	@property
	def device(self):
		return next(self.parameters()).device
	
	def forward(self, x, c):
		c = c.view(c.size(0), 1, 1)
		c = c.repeat(1, x.size(1), x.size(2))
		x = torch.cat([x,c], dim=1)
		return x

def ConvBlck(in_chanels, out_chanels, *arg, **kwargs):
	return torch.nn.Sequential(
		torch.nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=2, padding=1, bias=False),
		torch.nn.BatchNorm2d(num_features=out_chanels),
		torch.nn.LeakyReLU(negative_slope=0.2)
	)

def ConvTBlck(in_chanels, out_chanels, *arg, **kwargs):
	return torch.nn.Sequential(
		torch.nn.ConvTranspose2d(in_chanels, out_chanels, kernel_size=3, stride=2, padding=1, output_padding=1),
		torch.nn.BatchNorm2d(num_features=out_chanels),
		torch.nn.LeakyReLU(negative_slope=0.2)
	)

class ResidualBlck(torch.nn.Module):
	def __init__(self):
		self.blcks = torch.nn.Sequential(
			conv_blck(in_chanels=3+self.C, out_chanels=64),
			torch.nn.Conv2d(),
			torch.nn.BatchNorm2d()
		)
	
	def forward(self, x):
		x += self.blcks(x)
		return x

class USGenerator(torch.nn.Module):
	def __init__(self, C):
		self.C = C
		
		self.blcks = torch.nn.Sequential(
			InputExtractor(),
			ConvBlck(in_chanels=3+C, out_chanels=64), #in: DxDx(3+C) --> DxDx64 :out
			ConvBlck(in_chanels=64, out_chanels=128),      #in: DxDx64 --> D/2xD/2x128 :out
			ConvBlck(in_chanels=128, out_chanels=256),     #in: D/2xD/2x128 --> D/4xD/4x256 :out
			ResidualBlck(in_chanels=256, out_chanels=256), #in: D/4xD/4x256 === D/4xD/4x256 :out
			ConvTBlck(in_chanels=256, out_chanels=128),    #in: D/4xD/4x256 --> D/2xD/2x128 :out
			ConvTBlck(in_chanels=128, out_chanels=64),     #in: D/2xD/2x128 --> DxDx64 :out
			ConvTBlck(in_chanels=64, out_chanels=3)         #in: DxDx64 --> DxDx3 :out
		)
	
	def forward(self, x, c):
		x += self.blcks(x)
		return x

class ResnetBlck(torch.nn.Module):
	def __init__(self):
		self.blcks = torch.nn.Sequential(
			torch.nn.ReLU(),
			torch.nn.Conv2d(),
			torch.nn.ReLU(),
			torch.nn.Conv2d()
		)
	
	def forward(self, x):
		x += self.blcks(x)
		return x

class USDiscriminator(torch.nn.Module):
	def __init__(self, resnet_count=3):
		
		resnet_list = [
			ResnetBlck()
		]
		
		for i in range(resnet_count):
			resnet_list += [
				torch.nn.AvgPool2d(),
				ResnetBlck()
			]
		
		resnet_list += torch.nn.ReLU()
		
		self.conv = torch.nn.Conv2d()
		self.resnetblcks = torch.nn.Sequential(*resnet_list)
		self.f = torch.nn.Linear()
	
	def forward(self, x):
		out = self.conv(x)
		out = self.resnetblcks(x)
		out = self.f(x)
		
		return out

class GAN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		self.discriminator = USDiscriminator()
		seld.generator = USGenerator()
	
	def gen_fake(self, batch):

	def generator_loss(self, batch):
		loss = torch.nn.BCELoss()
		
		batch_size = batch.shape[0]
		y = torch.ones(batch_size).to(self.device)
		y_pred = self.discriminator(self.gen_fake(batch))
		
		return loss(y_pred, y)
		
	def discriminator_loss(self, batch_gen, batch_real):
		loss = torch.nn.BCELoss()
		
		batch_gen_size = batch_gen.shape[0]
		batch_real_size = batch_real.shape[0]
		
		X_fake = self.gen_fake(batch_gen)
		X_real = batch_real
		
		y_fake_pred = self.discriminator(X_fake)
		y_real_pred = seld.discriminator(X_real)
		
		y_fake = torch.zeroes(batch_gen_size)
		y_real = torch.ones(batch_real_size)
		
		return 0.5*loss(y_fake_pred, y_fake) + 0.5*loss(y_real_pred, y_real)