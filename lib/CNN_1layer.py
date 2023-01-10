import torch
import torch.nn as nn
from torch.autograd import Variable
########################### note ############################
# torch 0.4: https://pytorch.org/2018/04/22/0_4_0-migration-guide.html
#############################################################

class CNN(nn.Module):
	def __init__(self, conv_k_size = 512, conv_step = 256, filter_num = 100, adaptive_size=1, dropout = 0.3, type_number = 5):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(  
			nn.Conv1d(
				in_channels=1,					#input dimension (1,1,length)
				out_channels=filter_num,				#number of filters (1,200,length)
				kernel_size=conv_k_size,	#filter size  
				stride=conv_step,			#filter step size
				padding=conv_k_size-conv_step,	#padding for filtering
			),
			nn.ReLU(),
			# nn.AdaptiveMaxPool1d(output_size = adaptive_size),
			nn.AdaptiveAvgPool1d(output_size = adaptive_size),
			nn.Dropout(p=dropout)
		)
		self.fullconnect = nn.Linear(filter_num*adaptive_size,type_number)#100(conv2 output)*54(conv2input dim/MaxPool kernelsize), 4 for type of emotion
		self.softmax=nn.LogSoftmax()
		
	def forward(self,x):
		x = self.conv1(x)
		x = x.view(x.size(0),-1)	#size(0)=batch_size, -1 the output shape of MaxPool1d times together
		output = self.fullconnect(x)
		# output = self.softmax(x)
		return x, output
