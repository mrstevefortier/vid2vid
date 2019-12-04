### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
from options.test_options import ServeOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from PIL import Image
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

import falcon
import png
from wsgiref import simple_server

opt = ServeOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
if opt.dataset_mode == 'temporal':
    opt.dataset_mode = 'test'

def crop(Ai):
    w = Ai.size()[2]
    base = 32
    x_cen = w // 2
    bs = int(w * 0.5) // base * base
    return Ai[:,:,(x_cen-bs):(x_cen+bs)]

def get_image(json, size, params, input_type):
    if input_type != 'openpose':
        # Create image of size size
        A_img = 0 #Image.open(A_path).convert('RGB')
    else:            
        random_drop_prob = 0
        A_img = Image.fromarray(read_keypoints_from_text(json, size, random_drop_prob, opt.remove_face_labels, opt.basic_point_only))            

    is_img = input_type == 'img'
    method = Image.BICUBIC if is_img else Image.NEAREST
    transform_scaleA = get_transform(opt, params, method=method)
    A_scaled = transform_scaleA(A_img)
    return A_scaled

def get_img_params(opt, size):
    w, h = size
    new_h, new_w = h, w
    if 'resize' in opt.resize_or_crop:   # resize image to be loadSize x loadSize
        new_h = new_w = opt.loadSize            
    elif 'scaleWidth' in opt.resize_or_crop: # scale image width to be loadSize
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w
    elif 'scaleHeight' in opt.resize_or_crop: # scale image height to be loadSize
        new_h = opt.loadSize
        new_w = opt.loadSize * w // h
    elif 'randomScaleWidth' in opt.resize_or_crop:  # randomly scale image width to be somewhere between loadSize and fineSize
        new_w = random.randint(opt.fineSize, opt.loadSize + 1)
        new_h = new_w * h // w
    elif 'randomScaleHeight' in opt.resize_or_crop: # randomly scale image height to be somewhere between loadSize and fineSize
        new_h = random.randint(opt.fineSize, opt.loadSize + 1)
        new_w = new_h * w // h
    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4    

    crop_x = crop_y = 0
    crop_w = crop_h = 0
    if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
        if 'crop' in opt.resize_or_crop:      # crop patches of size fineSize x fineSize
            crop_w = crop_h = opt.fineSize
        else:
            if 'Width' in opt.resize_or_crop: # crop patches of width fineSize
                crop_w = opt.fineSize
                crop_h = opt.fineSize * h // w
            else:                              # crop patches of height fineSize
                crop_h = opt.fineSize
                crop_w = opt.fineSize * w // h

        crop_w, crop_h = make_power_2(crop_w), make_power_2(crop_h)        
        x_span = (new_w - crop_w) // 2
        #crop_x = np.maximum(0, np.minimum(x_span*2, int(np.random.randn() * x_span/3 + x_span)))        
        #crop_y = random.randint(0, np.minimum(np.maximum(0, new_h - crop_h), new_h // 8))
        crop_x = random.randint(0, np.maximum(0, new_w - crop_w))
        crop_y = random.randint(0, np.maximum(0, new_h - crop_h))        
    else:
        new_w, new_h = make_power_2(new_w), make_power_2(new_h)

    flip = (random.random() > 0.5) and (opt.dataset_mode != 'pose')
    return {'new_size': (new_w, new_h), 'crop_size': (crop_w, crop_h), 'crop_pos': (crop_x, crop_y), 'flip': flip}


def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:
            A = A[c:]
        A = torch.cat([A, Ai])
    return A

class Inference(object):
    def __init__(self):
        self.dataset = []
        self.model = create_model(opt)
        self.input_nc = 1 if opt.label_nc != 0 else opt.input_nc
        self.width = 0
        self.height = 0

    def createNewData(self, depth, width, height, json):
        size = # Create size from width/height
        params = get_img_params(self.opt, size)
        Ai = get_image(json, size, params, input_type='openpose')
        Bi = get_image(json, size, params, input_type='img')
        Ai, Bi = crop(Ai), crop(Bi) # only crop the central half region to save time
        return {'Ai': Ai, 'Bi': Bi}

    def on_get(self, req, resp):
        sizeString = req.get_param("size", required=True)
        sizeStringArray = sizeString.split(',')
        width = int(sizeStringArray[0])
        height = int(sizeStringArray[1])        
        if width != self.width || height != self.height:
            self.model.fake_B_prev = None
            self.width = width
            self.height = height

        # Read json
        json = "" # req.something

        newData = createNewData(self.dataset, opt.n_frames_G, width, height, json)
        self.dataset.append(newData)
        if (len(self.dataset) > opt.n_frames_G)
            self.dataset = self.dataset[opt.n_frames_G:]

        if len(self.dataset) == opt.n_frames_G:
            dataA = self.dataset[0].Ai
            dataB = self.dataset[0].Bi
            for i in [1, len(self.dataset)]:
                dataA = concat_frame(dataA, self.dataset[i].Ai, opt.n_frames_G:)
                dataB = concat_frame(dataB, self.dataset[i].Bi, opt.n_frames_G:)

            A = Variable(dataA).view(1, -1, self.input_nc, height, width)
            B = Variable(dataB).view(1, -1, opt.output_nc, height, width) if len(dataB.size()) > 2 else None
            inst = Variable(data['inst']).view(1, -1, 1, height, width) if len(data['inst'].size()) > 2 else None
            generated = self.model.inference(A, B, inst)
                
            image = util.tensor2im(generated[0].data[0])
            resp.body = image #pngFile.read()
            resp.content_type = falcon.MEDIA_PNG
            resp.status = falcon.HTTP_200

        else:
            image = 0 # empty image
            resp.body = image #pngFile.read()
            resp.content_type = falcon.MEDIA_PNG
            resp.status = falcon.HTTP_200                        

        
class Server:
    def __init__(self, port):
        self.port = port

    def serve(self):
        ## Real thing to call
        app = falcon.API()
        app.req_options.auto_parse_form_urlencoded=True

        inference = Inference()
        app.add_route('/infer', inference)

        print("Serving on port {}".format(self.port))
        httpd = simple_server.make_server('127.0.0.1', self.port, app)
        httpd.serve_forever()


server = Server(opt.port)
server.serve()
