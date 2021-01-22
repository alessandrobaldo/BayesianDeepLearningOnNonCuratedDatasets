import torch
			
def corruptCIFAR10Images(images, labels, non_curated):
	for i,image in enumerate(images):
		if torch.rand(1) <= non_curated:
			images[i] = image + torch.randn(image.size())
	return images, labels
			
def corruptCIFAR10Labels(images, labels, non_curated):
	for i,label in enumerate(labels):
		if torch.rand(1) <= non_curated:
			labels[i] = torch.randint(0,10,(1,1))
	return images, labels

def corruptONP(data, labels, non_curated):
	for i,label in enumerate(labels):
		if torch.rand(1) <= non_curated:
			if label == 0:
				labels[i] = 1
			else:
				labels[i] = 0
	return data, labels