import torch
import torch.nn as nn
import os, datetime, math, tgt, random, sys, time
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
from lib.CNN_1layer import CNN
# from lib.LSTM_attention import Seq2Seq_attention_BLSTM
from lib.DoubledCellLSTM2_NTN_2fc import LSTMModel
from lib.audio import trim_audio
# from cntn_model.NTN_model import NTN
from nltk.translate import bleu_score as bleu
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
SAVE_DATE = '0716'
SAVE_NAME = SAVE_DATE + 'AVcellcouple_allmean_NTNweight'
SAVE_FOLD = False
USE_CUDA = True
USE_pickle = True
# ====data========================================
VIDEO_TAG_PATH = './data/AFEW/train_AVtag/'
VIDEO_PATH = './data/AFEW/train_AV_mp4/'
WAV_PATH = './data/AFEW/train_au/'

VIDEO_VAL_TAG_PATH = './data/AFEW/Val_AFEW/Val_tag/'
VIDEO_VAL_PATH = './data/AFEW/Val_AFEW/Val_AV/'
WAV_VAL_PATH = './data/AFEW/Val_AFEW/Val_au/'
# ====small data==================================
# VIDEO_TAG_PATH = root + 'data/small_AVtag/'
# VIDEO_PATH = root + 'data/small_AV/'
# WAV_PATH = root + 'data/small_wav/'
# ================================================
MODEL_PATH = './model/'
RESULT_PATH = './result/AFEW/'

# ==== old model =========================================================
sound_model = '05215sound_pool2_filter100_total.pt'
emotion_model = '0616Conv_K256_7emo_pool1_filter100_total.pt'
# ==== new model =========================================================
sound_model = '0714_BAUMNNIME5sound_kernel32_filter700_total.pt'
emotion_model = '0621_7emo_kernel64_filter300_total.pt'
VGG_model = '7_15_AFEWmp4_72.t7'
RESULT_FILE = '_seq2seq_result.txt'
EMOTION_TYPE = {0:'anger', 1:'fear', 2:'sadness', 3:'surprise', 4:'neutral', 5:'boredom', 6:'happiness'}
SOUND_TYPE = {0:'laugh', 1:'breath', 2:'shout', 3:'silence', 4:'verbal'}
face_detect_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
# Hyper Parameters
FOLD, EPOCH, BATCH_SIZE = 1, 100, 100
SND_CNN_FILTER, SND_CNN_POOL = 700, 1
EMO_CNN_FILTER, EMO_CNN_POOL = 300, 1
LR = 0.01
INPUT_DIM = SND_CNN_POOL*SND_CNN_FILTER + EMO_CNN_POOL*EMO_CNN_FILTER
# HIDDEN_DIM = [512, 256, 128, 64, 32]
HIDDEN_DIM = [16,32, 64, 128, 256, 512]
LAYERS_DIM = 1
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
def write_result(result_path, result_file, fold_acc, fold_loss, fold_bleu, fold_conf, epoch, batch, lr, tag, model, run_time):
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
		# wf.write('mean of bleu: '+str(fold_bleu[i])+'\n')
	wf.write('mean of accuracy:'+str(sum(fold_acc)/len(fold_acc))+' | loss: '+str(sum(fold_loss)/len(fold_acc))+'\n')
	wf.close()

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
# device = torch.device('cpu')
cnn_sound = torch.load(MODEL_PATH + sound_model).to(device)
cnn_emotion = torch.load(MODEL_PATH + emotion_model).to(device)

cnn_sound.eval()
cnn_emotion.eval()
tags, val_tags = [], []
all_result = []

if USE_pickle:
	f = open('./feature/0716AFEWTRAINvisual_seqs_7_15_AFEWmp4_72.pkl', 'rb')
	visual_seqs = pickle.load(f)
	f.close()
	f = open('./feature/New_tag0716AFEWTRAINspeech_emotion_seqs_0621_7emo_kernel64_filter300_total.pkl', 'rb')
	speech_emotion_seqs = pickle.load(f)
	f.close()
	f = open('./feature/New_tag0716AFEWTRAINseqs_0621_7emo_kernel64_filter300_total_0714_BAUMNNIME5sound_kernel32_filter700_total.pkl', 'rb')
	seqs = pickle.load(f)
	f.close()
	f = open('./feature/0716AFEWTRAINimg_feature7_15_AFEWmp4_72.pkl', 'rb')
	img_feature = pickle.load(f)
	f.close()
	f = open('./feature/0716AFEWVALvisual_seqs_7_15_AFEWmp4_72.pkl', 'rb')
	val_visual_seqs = pickle.load(f)
	f.close()
	f = open('./feature/New_tag0716AFEWVALspeech_emotion_seqs_0621_7emo_kernel64_filter300_total.pkl', 'rb')
	val_speech_emotion_seqs = pickle.load(f)
	f.close()
	f = open('./feature/New_tag0716AFEWVALseqs_0621_7emo_kernel64_filter300_total_0714_BAUMNNIME5sound_kernel32_filter700_total.pkl', 'rb')
	val_seqs = pickle.load(f)
	f.close()
	f = open('./feature/0716AFEWVALimg_feature7_15_AFEWmp4_72.pkl', 'rb')
	val_img_feature = pickle.load(f)
	f.close()
	for video in sorted(os.listdir(VIDEO_PATH)):
		if video.endswith('mp4'):
			tag_file = video[:-3] + 'TextGrid'
			tg = tgt.read_textgrid(VIDEO_TAG_PATH + tag_file)
			tag_tier = tg.get_tier_by_name('silences')
			tag = np.array([[int(i.text[0])] for i in tag_tier])
			tags.append(tag[0])
	for video in sorted(os.listdir(VIDEO_VAL_PATH)):
		if video.endswith('mp4'):
			tag_file = video[:-3] + 'TextGrid'
			tg = tgt.read_textgrid(VIDEO_VAL_TAG_PATH + tag_file)
			tag_tier = tg.get_tier_by_name('silences')
			tag = np.array([[int(i.text[0])] for i in tag_tier])
			val_tags.append(tag[0])
	
else:
	print('get VGG feature...')
	cut_size = 44
	transform_test = transforms.Compose([
		transforms.FiveCrop(cut_size),
		transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	])
	detector = dlib.get_frontal_face_detector()
	net = VGG('VGG19')
	checkpoint = torch.load(os.path.join('FER2013_VGG19', VGG_model))
	net.load_state_dict(checkpoint['net'])
	net.to(device)
	net.eval()
	img_feature, visual_seqs = [], []
	total_bar = len(os.listdir(VIDEO_PATH))
	bar_num = 0
	wavs = []
	type_dis = [0]*len(EMOTION_TYPE)
	VGG_start_time = datetime.datetime.now()
	# for video in sorted(os.listdir(VIDEO_PATH)):
	# 	if video.endswith('mp4'):
	# 		print(video,'...')
	# 		tag_file = video[:-3] + 'TextGrid'
	# 		tg = tgt.read_textgrid(VIDEO_TAG_PATH + tag_file)
	# 		tag_tier = tg.get_tier_by_name('silences')
	# 		boundary = [i.start_time for i in tag_tier]
	# 		boundary.append(tag_tier[-1].end_time)	
	# 		tag = np.array([[int(i.text[0])] for i in tag_tier])
	# 		pph_audio = trim_audio(WAV_PATH + video[:-3]+'wav', boundary)
	# 		for i in tag:
	# 			type_dis[i[0]] += 1
	# 		tags.append(tag)
	# 		wavs.append(pph_audio)

	# 		vc = cv2.VideoCapture(VIDEO_PATH + video)
	# 		frame_count = 0 
	# 		all_frames = [] 
	# 		while(vc.isOpened()): 
	# 			ret, frame = vc.read() 
	# 			if ret is False: 
	# 				break 
	# 			all_frames.append(frame)
	# 			frame_count = frame_count + 1

	# 		# print('total frame is ' + str(frame_count) + ' and the file is ' + video)			
	# 		vc.release()
	# 		file_seg_img = []
	# 		for segment_index in range(len(boundary)-1):
	# 			print(video,segment_index,'finding...')
	# 			center_3_frame = []
	# 			num_of_frame = 0
	# 			for seg_frame in range(int((boundary[segment_index + 1] - boundary[segment_index])* (frame_count / boundary[-1]))-1):
	# 				bias = 0
	# 				catch = 0
	# 				over = 0
	# 				no_frame = 0
	# 				finding_time = time.time()
	# 				while catch == 0:
	# 					finding = time.time()
	# 					if finding - finding_time >= 10:
	# 						print('no frame to catch')
	# 						no_frame = 1
	# 						break
	# 					try:
	# 						if over == 0:
	# 							cap_frame_index = int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame + bias
	# 							img = all_frames[cap_frame_index]
	# 						else:
	# 							cap_frame_index = int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame - bias
	# 							img = all_frames[cap_frame_index]
	# 					except:
	# 						over = 1
	# 						bias = 0
	# 						cap_frame_index = int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame - bias
	# 						img = all_frames[cap_frame_index]

						
	# 					# Dlib
	# 					# resize_img = imutils.resize(img, width=1280)
	# 					# face_rects = detector(resize_img, 0)
	# 					# for i, d in enumerate(face_rects):
	# 					# 	x1 = d.left()
	# 					# 	y1 = d.top()
	# 					# 	x2 = d.right()
	# 					# 	y2 = d.bottom()
	# 					# crop_img = resize_img[y1:y2, x1:x2]
	# 					bias += 1
	# 					#OpenCV 
	# 					(h, w) = img.shape[:2]
	# 					blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
	# 					face_detect_net.setInput(blob)
	# 					detections = face_detect_net.forward()	
	# 					face = []	
	# 					for i in range(0, detections.shape[2]):
	# 						confidence = detections[0, 0, i, 2]
	# 						if confidence > 0.5:
	# 							box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	# 							(x1, y1, x2, y2) = box.astype("int")
	# 							width = x2-x1
	# 							height = y2-y1
	# 							crop_img = img[y1:y2, int((x1+x2)/2-(height/2)):int((x1+x2)/2+(height/2))]
	# 							try:
	# 								if len(crop_img) == 0 or len(crop_img[0]) == 0:
	# 									catch = 0
	# 								else:
	# 									face.append(crop_img)
	# 									catch = 1
	# 							except:
	# 								catch = 0		
	# 				print(video,segment_index,int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame,cap_frame_index,frame_count)
					
	# 				if no_frame == 1:
	# 					break

	# 				print('got',len(crop_img),len(crop_img[0]))					
	# 				crop_img = max(face, key=len)
	# 				# imgname = video[:-4] + str(cap_frame_index) + '.jpg'
	# 				# cv2.imwrite('./crop/' + imgname , crop_img)
	# 				# raw_img = io.imread('./crop/'+ imgname )
	# 				# raw_img = opencv2skimage(crop_img)
	# 				gray = rgb2gray(crop_img)
	# 				gray = resize(gray, (72,72), mode='symmetric').astype(np.uint8)
	# 				img = gray[:, :, np.newaxis]
	# 				img = np.concatenate((img, img, img), axis=2)
	# 				img = Image.fromarray(img)  
	# 				# img.show()
	# 				# time.sleep(2)

	# 				inputs = transform_test(img)
	# 				# inputs = img
	# 				ncrops, c, h, w = np.shape(inputs)
	# 				inputs = inputs.view(-1, c, h, w)
	# 				inputs = inputs.to(device)
	# 				bottleneck, _ = net(inputs)
	# 				# bottleneck = bottleneck.cpu()

	# 				# outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
	# 				# score = F.softmax(outputs_avg)
	# 				# _, predicted = torch.max(outputs_avg.data, 0)
	# 				# a = int(predicted.cpu().numpy())
	# 				# print('file: ' + video + " seg: " + str(segment_index) + ' is ok')
	# 				bottleneck_avg = bottleneck.view(ncrops, -1).mean(0).detach()

	# 				del inputs, img
	# 				# center_3_frame.append(bottleneck_avg.detach().cpu().numpy())
	# 				center_3_frame.append(bottleneck_avg)
	# 				num_of_frame += 1
	# 				if cap_frame_index == frame_count:
	# 					break
	# 			center_3_frame = sum(center_3_frame) / num_of_frame
	# 			center_3_frame.detach_().to(device)
	# 			# file_seg_img.append(torch.FloatTensor(center_3_frame))
	# 			file_seg_img.append(center_3_frame)
	# 		# print(video,segment_index, 'ok!')
	# 		lstm_input = torch.cat(file_seg_img).view(len(file_seg_img),1,-1).float().to(device)
	# 		visual_seqs.append(lstm_input)
	# 		img_feature.append(lstm_input.cpu().detach().numpy())
	# 		print("VGG progress:{0}%".format(round((bar_num + 1) * 100 / total_bar))) #print(xxx, end = '\r')
	# 		bar_num += 1
	# VGG_end_time = datetime.datetime.now()
	
	# print('number of pph:', sum(type_dis), ' dis:', type_dis)
	# print('original number of turns:',len(os.listdir(VIDEO_TAG_PATH)))
	# print('number of turns:',len(wavs))

	# # img_feature = torch.FloatTensor(img_feature)
	# # img_feature.to(device)
	# # print(img_feature)
	# cnn_start = datetime.datetime.now()
	# print('get cnn feature...')
	# seqs = []
	# speech_emotion_seqs = []
	# with torch.no_grad():
	# 	for i in range(len(wavs)):
	# 		seq_hidden = []
	# 		speech_emotion_seqs_hidden = []
	# 		for j in range(len(wavs[i])):
	# 			wav_tensor = torch.from_numpy(wavs[i][j][np.newaxis,np.newaxis,:]).to(device)
	# 			hidden_emo, output_emo = cnn_emotion(wav_tensor)
	# 			hidden_snd, output_snd = cnn_sound(wav_tensor)
	# 			hidden = torch.cat((hidden_snd, hidden_emo),1)
	# 			hidden.detach_().to(device)
	# 			speech_emotion_seqs_hidden.append(hidden_emo.detach())
	# 			seq_hidden.append(hidden)
	# 		# input of shape (seq_len, batch, input_size)
	# 		lstm_input = torch.cat(seq_hidden).view(len(seq_hidden),1,-1).float().to(device)
	# 		speech_emo_seqs = torch.cat(speech_emotion_seqs_hidden).view(len(speech_emotion_seqs_hidden),1,-1).float().to(device)
	# 		seqs.append(lstm_input)	
	# 		speech_emotion_seqs.append(speech_emo_seqs.cpu().detach().numpy())	
	# f = open('./feature/'+ SAVE_DATE + 'AFEWTRAINvisual_seqs_'+VGG_model[:-2]+'pkl','wb')
	# pickle.dump(visual_seqs,f)
	# f.close()	
	# f = open('./feature/New_tag' + SAVE_DATE +'AFEWTRAINseqs_' + emotion_model[:-3] + '_' + sound_model[:-2] + 'pkl','wb')
	# pickle.dump(seqs,f)
	# f.close()	
	# f = open('./feature/'+ SAVE_DATE + 'AFEWTRAINimg_feature'+VGG_model[:-2]+'pkl','wb')
	# pickle.dump(img_feature,f)
	# f.close()
	# f = open('./feature/New_tag'+ SAVE_DATE + 'AFEWTRAINspeech_emotion_seqs_'+emotion_model[:-2]+'pkl','wb')
	# pickle.dump(speech_emotion_seqs,f)
	# f.close()
	# print('cnn cost:',datetime.datetime.now()-cnn_start)
	total_bar = len(os.listdir(VIDEO_VAL_PATH))
	val_img_feature, val_visual_seqs, val_wavs = [], [], []
	for video in sorted(os.listdir(VIDEO_VAL_PATH)):
		if video.endswith('mp4'):
			print(video,'...')
			tag_file = video[:-3] + 'TextGrid'
			tg = tgt.read_textgrid(VIDEO_VAL_TAG_PATH + tag_file)
			tag_tier = tg.get_tier_by_name('silences')
			boundary = [i.start_time for i in tag_tier]
			boundary.append(tag_tier[-1].end_time)	
			tag = np.array([[int(i.text[0])] for i in tag_tier])
			pph_audio = trim_audio(WAV_VAL_PATH + video[:-3]+'wav', boundary)
			for i in tag:
				type_dis[i[0]] += 1
			val_tags.append(tag)
			val_wavs.append(pph_audio)

			vc = cv2.VideoCapture(VIDEO_VAL_PATH + video)
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
			file_seg_img = []
			for segment_index in range(len(boundary)-1):
				print(video,segment_index,'finding...')
				center_3_frame = []
				num_of_frame = 0
				for seg_frame in range(int((boundary[segment_index + 1] - boundary[segment_index])* (frame_count / boundary[-1]))-1):
					bias = 0
					catch = 0
					over = 0
					no_frame = 0
					finding_time = time.time()
					while catch == 0:
						finding = time.time()
						if finding - finding_time >= 10:
							print('no frame to catch')
							no_frame = 1
							break
						try:
							if over == 0:
								cap_frame_index = int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame + bias
								img = all_frames[cap_frame_index]
							else:
								cap_frame_index = int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame - bias
								img = all_frames[cap_frame_index]
						except:
							over = 1
							bias = 0
							cap_frame_index = int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame - bias
							img = all_frames[cap_frame_index]

						
						# Dlib
						# resize_img = imutils.resize(img, width=1280)
						# face_rects = detector(resize_img, 0)
						# for i, d in enumerate(face_rects):
						# 	x1 = d.left()
						# 	y1 = d.top()
						# 	x2 = d.right()
						# 	y2 = d.bottom()
						# crop_img = resize_img[y1:y2, x1:x2]
						bias += 1
						#OpenCV 
						(h, w) = img.shape[:2]
						blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
						face_detect_net.setInput(blob)
						detections = face_detect_net.forward()	
						face = []	
						for i in range(0, detections.shape[2]):
							confidence = detections[0, 0, i, 2]
							if confidence > 0.5:
								box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
								(x1, y1, x2, y2) = box.astype("int")
								width = x2-x1
								height = y2-y1
								crop_img = img[y1:y2, int((x1+x2)/2-(height/2)):int((x1+x2)/2+(height/2))]
								
								try:
									
									if len(crop_img) == 0 or len(crop_img[0]) == 0:
										catch = 0
									else:
										face.append(crop_img)
										catch = 1
								except:
									catch = 0		
					print(video,segment_index,int((boundary[segment_index]) * (frame_count / boundary[-1])) + seg_frame,cap_frame_index,frame_count)
					
					if no_frame == 1:
						break
					print('got',len(crop_img),len(crop_img[0]))
					crop_img = max(face, key=len)
					# imgname = video[:-4] + str(cap_frame_index) + '.jpg'
					# cv2.imwrite('./crop/' + imgname , crop_img)
					# raw_img = io.imread('./crop/'+ imgname )
					# raw_img = opencv2skimage(crop_img)
					gray = rgb2gray(crop_img)
					gray = resize(gray, (72,72), mode='symmetric').astype(np.uint8)
					img = gray[:, :, np.newaxis]
					img = np.concatenate((img, img, img), axis=2)
					img = Image.fromarray(img)  
					# img.show()
					# time.sleep(2)

					inputs = transform_test(img)
					# inputs = img
					ncrops, c, h, w = np.shape(inputs)
					inputs = inputs.view(-1, c, h, w)
					inputs = inputs.to(device)
					bottleneck, _ = net(inputs)
					# bottleneck = bottleneck.cpu()

					# outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
					# score = F.softmax(outputs_avg)
					# _, predicted = torch.max(outputs_avg.data, 0)
					# a = int(predicted.cpu().numpy())
					# print('file: ' + video + " seg: " + str(segment_index) + ' is ok')
					bottleneck_avg = bottleneck.view(ncrops, -1).mean(0).detach()

					del inputs, img
					# center_3_frame.append(bottleneck_avg.detach().cpu().numpy())
					center_3_frame.append(bottleneck_avg)
					num_of_frame += 1
					if cap_frame_index == frame_count:
						break
				center_3_frame = sum(center_3_frame) / num_of_frame
				center_3_frame.detach_().to(device)
				# file_seg_img.append(torch.FloatTensor(center_3_frame))
				file_seg_img.append(center_3_frame)
			# print(video,segment_index, 'ok!')
			lstm_input = torch.cat(file_seg_img).view(len(file_seg_img),1,-1).float().to(device)
			val_visual_seqs.append(lstm_input)
			val_img_feature.append(lstm_input.cpu().detach().numpy())
			print("VGG progress:{0}%".format(round((bar_num + 1) * 100 / total_bar))) #print(xxx, end = '\r')
			bar_num += 1
	VGG_end_time = datetime.datetime.now()
	
	print('original number of turns:',len(os.listdir(VIDEO_VAL_TAG_PATH)))
	print('number of turns:',len(val_wavs))

	# img_feature = torch.FloatTensor(img_feature)
	# img_feature.to(device)
	# print(img_feature)
	cnn_start = datetime.datetime.now()
	print('get cnn feature...')
	val_seqs = []
	val_speech_emotion_seqs = []
	with torch.no_grad():
		for i in range(len(val_wavs)):
			seq_hidden = []
			speech_emotion_seqs_hidden = []
			for j in range(len(val_wavs[i])):
				wav_tensor = torch.from_numpy(val_wavs[i][j][np.newaxis,np.newaxis,:]).to(device)
				hidden_emo, output_emo = cnn_emotion(wav_tensor)
				hidden_snd, output_snd = cnn_sound(wav_tensor)
				hidden = torch.cat((hidden_snd, hidden_emo),1)
				hidden.detach_().to(device)
				speech_emotion_seqs_hidden.append(hidden_emo.detach())
				seq_hidden.append(hidden)
			# input of shape (seq_len, batch, input_size)
			lstm_input = torch.cat(seq_hidden).view(len(seq_hidden),1,-1).float().to(device)
			speech_emo_seqs = torch.cat(speech_emotion_seqs_hidden).view(len(speech_emotion_seqs_hidden),1,-1).float().to(device)
			val_seqs.append(lstm_input)	
			val_speech_emotion_seqs.append(speech_emo_seqs.cpu().detach().numpy())	
	f = open('./feature/'+ SAVE_DATE + 'AFEWVALvisual_seqs_'+VGG_model[:-2]+'pkl','wb')
	pickle.dump(val_visual_seqs,f)
	f.close()	
	f = open('./feature/New_tag' + SAVE_DATE +'AFEWVALseqs_' + emotion_model[:-3] + '_' + sound_model[:-2] + 'pkl','wb')
	pickle.dump(val_seqs,f)
	f.close()	
	f = open('./feature/'+ SAVE_DATE + 'AFEWVALimg_feature'+VGG_model[:-2]+'pkl','wb')
	pickle.dump(val_img_feature,f)
	f.close()
	f = open('./feature/New_tag'+ SAVE_DATE + 'AFEWVALspeech_emotion_seqs_'+emotion_model[:-2]+'pkl','wb')
	pickle.dump(val_speech_emotion_seqs,f)
	f.close()
	print('cnn cost:',datetime.datetime.now()-cnn_start)
smooth = bleu.SmoothingFunction()

for tensor_dim in range(1,6):
	NTN_model = '0714_cntn_T' + str(tensor_dim) + '.pkl'
	NTN = torch.load(MODEL_PATH + NTN_model).to(device)
	NTN.eval()
	for hidden_i in range(len(HIDDEN_DIM)):
		fold_acc, fold_loss, fold_bleu = [0]*FOLD, [0]*FOLD, [0]*FOLD
		fold_conf = []
		for fold in range(FOLD):
			train_start = datetime.datetime.now()
			lstm = LSTMModel(input_dim1 = INPUT_DIM, input_dim2 = 512, hidden_dim = HIDDEN_DIM[hidden_i], half_hidden_dim = int(HIDDEN_DIM[hidden_i]/2), layer_dim = LAYERS_DIM, output_dim = len(EMOTION_TYPE.keys()))
			lstm.float().to(device)
			optimizer = torch.optim.Adam(lstm.parameters(), lr = LR)
			# loss_func = nn.NLLLoss()
			loss_func = nn.CrossEntropyLoss()
			train_seq_audio, train_seq_visual, train_tag, train_type, train_seq_NTN_speech_emo, train_seq_NTN_visual_emo = [], [], [], [], [], []
			test_seq_audio, test_seq_visual, test_tag, test_type, test_seq_NTN_speech_emo, test_seq_NTN_visual_emo = [], [], [], [], [], []
			# for item in range(len(seqs)):
			# 	if item % 5 == fold:
			# 		test_seq_audio.append(seqs[item])
			# 		test_seq_visual.append(visual_seqs[item])
			# 		test_tag.append(tags[item])
			# 		test_seq_NTN_speech_emo.append(speech_emotion_seqs[item])
			# 		test_seq_NTN_visual_emo.append(img_feature[item])
			# 	else:
			# 		train_seq_audio.append(seqs[item])
			# 		train_seq_visual.append(visual_seqs[item])
			# 		train_tag.append(tags[item])
			# 		train_seq_NTN_speech_emo.append(speech_emotion_seqs[item])
			# 		train_seq_NTN_visual_emo.append(img_feature[item])
			train_seq_audio = seqs
			train_seq_NTN_speech_emo = speech_emotion_seqs
			train_seq_NTN_visual_emo = img_feature
			train_seq_visual = visual_seqs
			train_tag = tags

			test_seq_audio = val_seqs
			test_seq_NTN_speech_emo = val_speech_emotion_seqs
			test_seq_NTN_visual_emo = val_img_feature
			test_seq_visual = val_visual_seqs	
			test_tag = val_tags
			
			
			for epoch in range(EPOCH):
				result = []
				batchi = 0
				lastloss = 0
				confusion_table = np.zeros([len(EMOTION_TYPE.keys()) + 1, len(EMOTION_TYPE.keys())])

				for i in range(len(train_seq_audio)):
					score =[]			
					# print(train_seq_audio[i])
					# print(len(train_seq_audio[i]))
					# print(train_seq_visual[i])
					# print(len(train_seq_visual[i]))
					for j in range(len(train_seq_audio[i])):
						NTN_speech_emo = torch.FloatTensor(train_seq_NTN_speech_emo[i][j]).to(device)
						NTN_visual_emo = torch.FloatTensor(train_seq_NTN_visual_emo[i][j]).to(device)
						NTN_out = NTN(NTN_speech_emo, NTN_visual_emo)
						score.append(NTN_out.cpu().detach().numpy()[0][1])
					de_out = lstm(train_seq_audio[i], train_seq_visual[i], score, device).to(device)
					# h_n of lstm(num_layers * num_directions, batch, hidden_size) = (1,1,8)
					result.append(de_out[-1])
		
					if (len(result)==BATCH_SIZE)|(i+1==len(train_seq_audio)):
						losses = []
						for index in range(i+1-batchi*BATCH_SIZE):
							tag_va = torch.from_numpy(train_tag[batchi*BATCH_SIZE+index]).view(-1).to(device)
							# loss = loss_func(result[index].view(result[index].size(0),-1), tag_va)
							# print(result[index].view(1,7), tag_va)
							loss = loss_func(result[index].view(1,7), tag_va)							
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
					score =[]
					for j in range(len(test_seq_audio[i])):
						NTN_speech_emo = torch.FloatTensor(test_seq_NTN_speech_emo[i][j]).to(device)
						NTN_visual_emo = torch.FloatTensor(test_seq_NTN_visual_emo[i][j]).to(device)
						NTN_out = NTN(NTN_speech_emo, NTN_visual_emo)
						score.append(NTN_out.cpu().detach().numpy()[0][1])
					de_out = lstm(test_seq_audio[i], test_seq_visual[i], score, device).to(device)
					de_out = de_out[-1]
					pred_tag = torch.max(de_out.view(1,7),1)[1].to(device)
					# print(pred_tag)
					for ind, v in enumerate(pred_tag):
						# print(v.item(),test_tag[i][ind],'tag: ',i)
						result_pre.append(v.item())
						conf_tag.append(test_tag[i][ind])
						if v.item() == test_tag[i][ind]:
							acc_count += 1
						# confusion_table[v.item(), test_tag[i][ind]] += 1
				# for i in range(len(ne_si_seqs)):
				# 	de_out, de_h_n, weight = lstm(ne_si_seqs[i], device)
				# 	pred_tag = torch.max(de_out.view(de_out.size(0),-1),1)[1].to(device).view(de_out.size(0),-1)
				# 	for ind, v in enumerate(pred_tag):
				# 		result_pre.append(v.item())
				# 		conf_tag.append(ne_si_tags[i][ind])
				# 		if v.item()==ne_si_tags[i][ind]:
				# 			acc_count += 1
				# for i in range(len(EMOTION_TYPE.keys())):
				# 	confusion_table[len(EMOTION_TYPE.keys()),i] = round(confusion_table[i,i] / sum(confusion_table[:len(EMOTION_TYPE.keys()),i]),2)
				acc = acc_count/(len(conf_tag))
				confusion = confusion_matrix(conf_tag,result_pre)
				# print('ffmpeg 10db')
				# print('whole training train with bleu and ne_si as test')
				print(confusion)
				# print(confusion_table)
				print('Tensor',tensor_dim)
				print('whole:Hidden:',HIDDEN_DIM[hidden_i],'Fold:',fold,' Epoch:', epoch,'| train loss: %.6f' % lastloss, '| test accuracy: %.4f ' % acc)
			
			bleus = []
			for i in range(len(test_seq_audio)):
				score =[]			
				for j in range(len(test_seq_audio[i])):
					NTN_speech_emo = torch.FloatTensor(test_seq_NTN_speech_emo[i][j]).to(device)
					NTN_visual_emo = torch.FloatTensor(test_seq_NTN_visual_emo[i][j]).to(device)
					NTN_out = NTN(NTN_speech_emo, NTN_visual_emo)
					score.append(NTN_out.cpu().detach().numpy()[0][1])
				de_out = lstm(test_seq_audio[i], test_seq_visual[i], score, device).to(device)
				pred_tag = torch.max(de_out.view(de_out.size(0),-1),1)[1].to(device).view(de_out.size(0),-1)
				bleu_hyp = [EMOTION_TYPE[tag.item()] for tag in pred_tag]
				# bleu_ref = [EMOTION_TYPE[tag[0]] for tag in test_tag[i]]
				# bleus.append(100*bleu.sentence_bleu([bleu_ref], bleu_hyp, smoothing_function = None, weights = (0.4, 0.4, 0.2, 0)))
				# bleus.append(100*bleu.sentence_bleu([bleu_ref], bleu_hyp, smoothing_function = smooth.method2, weights = (0.4, 0.4, 0.2, 0)))
			print('mean of bleu:',sum(bleus)/len(bleus))
			if SAVE_FOLD:
				model_fold = SAVE_NAME+'_SNDp'+str(SND_CNN_POOL)+'f'+str(SND_CNN_FILTER)+'_EMOp'+str(EMO_CNN_POOL)+'f'+str(EMO_CNN_FILTER)+'_hidden'+str(HIDDEN_DIM[hidden_i])+'_tensor'+str(tensor_dim)+'_fold'+str(fold)+'.pt'
				torch.save(lstm,MODEL_PATH+model_fold)
			fold_acc[fold] = acc
			fold_loss[fold] = lastloss
			# fold_bleu[fold] = sum(bleus)/len(bleus)
			fold_conf.append(confusion)
			train_end = datetime.datetime.now()
			print(lstm)
			train_time = train_end-train_start
			print('training time:',train_time)
		txt_name = SAVE_NAME+'_SNDp'+str(SND_CNN_POOL)+'f'+str(SND_CNN_FILTER)+'_EMOp'+str(EMO_CNN_POOL)+'f'+str(EMO_CNN_FILTER)+'_hidden'+str(HIDDEN_DIM[hidden_i])+'_tensor'+str(tensor_dim)+RESULT_FILE
		write_result(RESULT_PATH, txt_name, fold_acc, fold_loss, fold_bleu, fold_conf, EPOCH, BATCH_SIZE, LR, EMOTION_TYPE, lstm, train_time)
		all_result.append([fold_acc, fold_loss, fold_conf, lstm, train_time, HIDDEN_DIM[hidden_i]])
		if np.mean(fold_acc) > BEST_ACC:
			BEST_ACC = np.mean(fold_acc)
			BEST_FILE = txt_name
			print(BEST_ACC,' ',BEST_FILE)
		if np.mean(fold_loss) < BEST_LOSS:
			BEST_LOSS = np.mean(fold_loss)
			BEST_LFILE = txt_name
	
end_time = datetime.datetime.now()

# wf = open('./'+SAVE_NAME+'_best.txt','w')
# wf.write('maximum accuracy file:'+BEST_FILE)
# wf.write('\naccuracy: '+str(BEST_ACC))
# wf.write('\nminimum loss file:'+BEST_LFILE)
# wf.write('\nloss: '+str(BEST_LOSS))
# wf.write('\ntraining time:'+str(end_time-begin_time))
# wf.write('\n')
# for i in all_result:
# 	wf.write('*********************************\n')
# 	wf.write('hidden size: '+str(i[-1])+' train time:'+str(i[-2])+'\n')
# 	wf.write('accuracy: '+str(i[0])+'\n')
# 	wf.write('loss:     '+str(i[1])+'\n')
# wf.close()