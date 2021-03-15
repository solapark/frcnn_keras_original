import os
import datetime
import csv
import re

def make_save_dir(base_dir, save_dir, reset):
	save_path = os.path.join(base_dir, 'experiment', save_dir)
	if(reset):
		model_path = os.path.join(save_path, 'model')

		if reset : os.system('rm -rf %s'%(save_path))

		os.makedirs(save_path, exist_ok = True)
		os.makedirs(model_path, exist_ok = True)
	return save_path

def write_config(path, option, C, is_reset):
	if(is_reset):
		f = open(path, 'w')
	else :
		f = open(path, 'a')
	f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
	for k, v in vars(C).items():
		f.write('{}: {}\n'.format(k, v))
	f.write('\n')
	for k, v in vars(option).items():
		f.write('{}: {}\n'.format(k, v))
	f.write('\n')
	f.close()

class Log_manager:
	def __init__(self, save_dir, reset):
		self.path = os.path.join(save_dir, 'log.csv')
		if(reset): self.write(['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'time'])
	
	def write(self, c):
		f = open(self.path, 'a')
		wr = csv.writer(f)
		wr.writerow(c)
		f.close()

class Model_path_manager:
	def __init__(self, model_dir, resume):
		self.model_dir = model_dir
		base_name = 'model.hdf5'
		self.name_pattern = "model_([0-9]*)\.hdf5"
		name_prefix, self.ext = base_name.split('.')

		if(self.ext != 'hdf5'):
			print('Output weights must have .hdf5 filetype')
			exit(1)
		self.prefix = os.path.join(self.model_dir, name_prefix)
		
		if resume :
			self.resume_epoch = self.get_resume_epoch()
			self.cur_epoch = self.resume_epoch + 1
		else :
			self.cur_epoch = 1
		
	def get_resume_epoch(self) :
		filenames = os.listdir(self.model_dir)
		epoch_list = []
		for filename in filenames :
			model_path_regex = re.match(self.name_pattern, filename)
			epoch_list.append(int(model_path_regex.group(1)))
		return max(epoch_list)

	def get_resume_path(self):
		return '%s_%04d.%s' %(self.prefix, self.resume_epoch, self.ext)

	def get_save_path(self):
		save_path = '%s_%04d.%s' %(self.prefix, self.cur_epoch, self.ext)
		self.cur_epoch += 1 
		return save_path
