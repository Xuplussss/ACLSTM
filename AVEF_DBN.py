import torch
import torch.nn as nn
import os, datetime, math, tgt, random, sys, time, librosa
import torch.nn.functional as F
import scipy.io.wavfile
import numpy as np
import cv2
import imutils
import pickle
import dlib
import transforms as transforms
from skimage import io
from skimage.transform import resize
from skimage.viewer import ImageViewer
from PIL import Image
from models import *
from sklearn.metrics import confusion_matrix
from lib.DBN import DBN
from lib.audio import trim_audio
# from cntn_model.NTN_model import NTN
random.seed(2708)
torch.manual_seed(2708)
np.random.seed(2708)
begin_time = datetime.datetime.now()
########################### note ############################
# torch 0.4: https://pytorch.org/2018/04/22/0_4_0-migration-guide.html
# EMOTION = {0:'anger', 1:'anxiety', 2:'sadness', 3:'surprise', 4:'neutral', 5:'boredom', 6:'happiness', 7:'silence'}
# SOUND = {0:'laugh', 1:'breath', 2:'shout', 3:'silence', 4:'verbal'}
#############################################################

# Enviroment Parameters
SAVE_DATE = '1016'
SAVE_NAME = SAVE_DATE + 'DBN'
SAVE_FOLD = False
USE_CUDA = True
USE_pickle = True

VIDEO_TAG_PATH = './data/all_av_tag/'
VIDEO_PATH = './data/AV/'
VIDEO_PATH = './data/small_AV/'
WAV_PATH = './data/BAUM_audio_mono_16k/'

# ================================================
MODEL_PATH = './model/'
RESULT_PATH = './result/'

# ==== now model =========================================================
audio_model = '107__AVEF_audio_CNN.pt'
visual_model = '813__AVEF_visual.pt'

RESULT_FILE = '_DBN_result.txt'
EMOTION_TYPE = {0:'anger', 1:'fear', 2:'sadness', 3:'surprise', 4:'neutral', 5:'boredom', 6:'happiness'}
KERNEL_SIZE = 11

# Hyper Parameters
FOLD, EPOCH, BATCH_SIZE = 1, 50, 100
LR = 0.01
INPUT_DIM = 2000
# Best Result Record
BEST_ACC = 0
BEST_FILE = ''
BEST_LOSS = 5
BEST_LFILE = ''
def opencv2skimage(src):
    src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    # src.astype(float)
    # src = src / 255
    return src
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def write_result(result_path, result_file, fold_acc, fold_loss, fold_conf, epoch, batch, lr, tag, model, run_time):
	# parameter, accuracy, loss, epoch, batch, trainingtime, lr, dropout, bestparameter
	wf = open(result_path+result_file, 'w')
	wf.write('model:\n')
	wf.write(str(model))
	wf.write('\n')
	wf.write('VGG: '+VGG_model+'\n')
	wf.write('audio emotion CNN: '+emotion_model+'\n')
	wf.write('audio sound type CNN: '+sound_model+'\n')
	wf.write('NTN: '+NTN_model+'\n')		
	wf.write('run time:'+str(run_time)+'\n')
	wf.write('epoch:   '+str(epoch)+'\n')
	wf.write('batch:   '+str(batch)+'\n')
	wf.write('learn rate:'+str(lr)+'\n')
	wf.write('tag type:'+str(tag)+'\n')
	for i in range(len(fold_acc)):
		wf.write(str(fold_conf[i])+'\n')
		wf.write('accuracy: '+str(fold_acc[i])+'| loss: '+str(fold_loss[i])+'\n')
	wf.write('mean of accuracy:'+str(sum(fold_acc)/len(fold_acc))+' | loss: '+str(sum(fold_loss)/len(fold_acc))+'\n')
	wf.close()

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
# device = torch.device('cpu')
Alex = torch.load(MODEL_PATH + audio_model).to(device)
C3D = torch.load(MODEL_PATH + visual_model).to(device)

Alex.eval()
C3D.eval()

tags = []
all_result = []

print('get face img ...')

detector = dlib.get_frontal_face_detector()

img_feature, visual_seqs = [], []
total_bar = len(os.listdir(VIDEO_PATH))
bar_num = 0
type_dis = [0]*len(EMOTION_TYPE)
img_segment = []
wavs_mel = [] 
segmentNUM = 0
for video in sorted(os.listdir(VIDEO_PATH)):
	if video.endswith('mp4'):
		tag_file = video[:-3] + 'TextGrid'
		tg = tgt.read_textgrid(VIDEO_TAG_PATH + tag_file)
		tag_tier = tg.get_tier_by_name('silences')
		boundary = [i.start_time for i in tag_tier]
		boundary.append(tag_tier[-1].end_time)	
		tag = np.array([[int(i.text[0])] for i in tag_tier])
		segmentNUM += len(boundary)-1
		vc = cv2.VideoCapture(VIDEO_PATH + video)
		frame_count = 0 
		all_frames = [] 
		while(vc.isOpened()): 
			ret, frame = vc.read() 
			if ret is False: 
				break 
			all_frames.append(frame)
			frame_count = frame_count + 1
		# print('total frame is ' + str(frame_count) + ' and the file is ' + video)			
		vc.release()
		if all_frames == []:
			continue
		wavfile = video[:-3] + 'wav'
		for segment_index in range(len(boundary)-1):
			record = 0
			noface = 0
			segment_frame = np.empty(0)			
			num_of_frame = 0
			for bias in np.linspace(-7, 8, 16, dtype=int): 			
				cap_frame_index = int(((boundary[segment_index+1] + boundary[segment_index]) / 2) * (frame_count / boundary[-1])) + bias
				# try:
				if cap_frame_index < boundary[segment_index] * (frame_count / boundary[-1]):
					cap_frame_index = int(boundary[segment_index] * (frame_count / boundary[-1]))
				elif cap_frame_index >= boundary[segment_index+1] * (frame_count / boundary[-1]):
					cap_frame_index = int(boundary[segment_index+1] * (frame_count / boundary[-1]))-1
				if cap_frame_index >= len(all_frames):
					cap_frame_index = len(all_frames) - 1
				elif cap_frame_index <= 0:
					cap_frame_index = 0
				print(video, segment_index, cap_frame_index, len(all_frames))
				
				img = all_frames[cap_frame_index]
				resize_img = imutils.resize(img, width=1280)
				face_rects = detector(resize_img, 0)
				for i, d in enumerate(face_rects):
					x1 = d.left()
					y1 = d.top()
					x2 = d.right()
					y2 = d.bottom()
				crop_img = resize_img[y1:y2, x1:x2]
				if len(crop_img) == 0:
					noface = 1
					break
				raw_img = opencv2skimage(crop_img)
				# except:
				# 	print('no face!')

				# print(cap_frame_index,boundary[segment_index] * (frame_count / boundary[-1]),boundary[segment_index+1] * (frame_count / boundary[-1]))
				
				gray = rgb2gray(raw_img)
				gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
				img = gray[:, :, np.newaxis]
				img = np.concatenate((img, img, img), axis=2)
				if num_of_frame == 0:
					segment_frame = img
				else:
					# np.vstack((segment_frame,img))
					segment_frame = np.concatenate((segment_frame, img))				
				num_of_frame += 1
			if noface == 1:
				break
			seg = np.reshape(segment_frame,(-1,3,48,48))
			if len(seg) != 16:
				print(video,'error!')
			du_time = boundary[segment_index+1] - boundary[segment_index]
			y, sr = librosa.load(WAV_PATH+wavfile, offset=boundary[segment_index], duration=du_time)
			D = np.abs(np.fft.rfft(y))**2
			S = librosa.feature.melspectrogram(D, sr=sr, n_mels=64, fmax=8000)
			if len(S[0]) >= KERNEL_SIZE :
				F1 = np.diff(S)
				for ii in range(len(F1)):
					if ii == 0:
						f1 = np.insert(np.zeros(1),0,F1[ii])
						ff = f1
					else:
						f1 = np.insert(np.zeros(1),0,F1[ii])
						ff = np.vstack((ff,f1))
				F2 = np.diff(ff)
				for ii in range(len(F2)):
					if ii == 0:
						f1 = np.insert(np.zeros(1),0,F2[ii])
						fff = f1
					else:
						f1 = np.insert(np.zeros(1),0,F2[ii])
						fff = np.vstack((fff,f1))
				mel = np.concatenate((S,ff,fff), axis=0).reshape((3,64,-1)) # np.vstack((S,ff,fff)).reshape((3,64,-1))
				record = 1
			if record == 1:
				img_segment.append(seg)
				tags.append(tag[segment_index])
				type_dis[int(tag[segment_index])] += 1
				wavs_mel.append(mel)
		print("VGG progress:{0}%".format(round((bar_num + 1) * 100 / total_bar)), end = '\r')
		bar_num += 1			
print(segmentNUM)
print(len(img_segment))
print(len(wavs_mel))
fold_acc, fold_loss = [0]*FOLD, [0]*FOLD
fold_conf = []
for fold in range(FOLD):
	train_start = datetime.datetime.now()
	dbn = DBN()
	dbn.float().to(device)
	optimizer = torch.optim.Adam(dbn.parameters(), lr = LR)
	# loss_func = nn.NLLLoss()
	loss_func = nn.CrossEntropyLoss()
	train_seq_audio, train_seq_visual, train_tag, train_type = [], [], [], []
	test_seq_audio, test_seq_visual, test_tag, test_type = [], [], [], []
	for item in range(len(seqs)):
		if item % 5 == fold:
			test_seq_audio.append(seqs[item])
			test_seq_visual.append(visual_seqs[item])
			test_tag.append(tags[item])
		else:
			train_seq_audio.append(seqs[item])
			train_seq_visual.append(visual_seqs[item])
			train_tag.append(tags[item])
			
	
	for epoch in range(EPOCH):
		result = []
		batchi = 0
		lastloss = 0
		confusion_table = np.zeros([len(EMOTION_TYPE.keys()) + 1, len(EMOTION_TYPE.keys())])

		for i in range(len(train_seq_audio)):
			train_seq = torch.cat((train_seq_audio[i],train_seq_visual[i]),1).float().to(device)
			de_out = dbn(train_seq).to(device)
			result.append(de_out)

			if (len(result)==BATCH_SIZE)|(i+1==len(train_seq_audio)):
				losses = []
				for index in range(i+1-batchi*BATCH_SIZE):
					tag_va = torch.from_numpy(train_tag[batchi*BATCH_SIZE+index]).view(-1).to(device)
					loss = loss_func(result[index].view(result[index].size(0),-1), tag_va)
					losses.append(loss.view(1))
				batchi += 1
				result = []
				optimizer.zero_grad()
				loss_batch = torch.cat(losses).mean().to(device)
				loss_batch.backward()
				optimizer.step()
				lastloss += loss_batch.item()
		lastloss = lastloss/math.ceil(len(train_seq_audio)/BATCH_SIZE)
		
		acc_count = 0
		result_pre = []
		conf_tag = []
		for i in range(len(test_seq_audio)):
			test_seq = torch.cat((test_seq_audio[i],test_seq_visual[i]),1).float().to(device)
			de_out = dbn(test_seq).to(device)
			pred_tag = torch.max(de_out.view(de_out.size(0),-1),1)[1].to(device).view(de_out.size(0),-1)
			for ind, v in enumerate(pred_tag):
				result_pre.append(v.item())
				conf_tag.append(test_tag[i][ind])
				if v.item() == test_tag[i][ind]:
					acc_count += 1
				confusion_table[v.item(), test_tag[i][ind]] += 1
		
		
		for i in range(len(EMOTION_TYPE.keys())):
			confusion_table[len(EMOTION_TYPE.keys()),i] = round(confusion_table[i,i] / sum(confusion_table[:len(EMOTION_TYPE.keys()),i]),2)
		acc = acc_count/(len(conf_tag))
		confusion = confusion_matrix(conf_tag,result_pre)
		# print('ffmpeg 10db')
		print(confusion)
		# print(confusion_table)
		print('Fold:',fold,' Epoch:', epoch,'| train loss: %.6f' % lastloss, '| test accuracy: %.4f ' % acc)
	
	
	if SAVE_FOLD:
		model_fold = SAVE_NAME+'_SNDp'+str(SND_CNN_POOL)+'f'+str(SND_CNN_FILTER)+'_EMOp'+str(EMO_CNN_POOL)+'f'+str(EMO_CNN_FILTER)+'_hidden'+str(HIDDEN_DIM[hidden_i])+'_fold'+str(fold)+'.pt'
		torch.save(dbn,MODEL_PATH+model_fold)
	fold_acc[fold] = acc
	fold_loss[fold] = lastloss
	fold_conf.append(confusion)
	train_end = datetime.datetime.now()
	print(dbn)
	train_time = train_end-train_start
	print('training time:',train_time)
txt_name = SAVE_NAME+'_SNDp'+str(SND_CNN_POOL)+'f'+str(SND_CNN_FILTER)+'_EMOp'+str(EMO_CNN_POOL)+'f'+str(EMO_CNN_FILTER)+'_hidden'+str(HIDDEN_DIM[hidden_i])+RESULT_FILE
write_result(RESULT_PATH, txt_name, fold_acc, fold_loss, fold_conf, EPOCH, BATCH_SIZE, LR, EMOTION_TYPE, dbn, train_time)
all_result.append([fold_acc, fold_loss, fold_conf, dbn, train_time, HIDDEN_DIM[hidden_i]])
if np.mean(fold_acc) > BEST_ACC:
	BEST_ACC = np.mean(fold_acc)
	BEST_FILE = txt_name
	print(BEST_ACC,' ',BEST_FILE)
if np.mean(fold_loss) < BEST_LOSS:
	BEST_LOSS = np.mean(fold_loss)
	BEST_LFILE = txt_name
	
end_time = datetime.datetime.now()
